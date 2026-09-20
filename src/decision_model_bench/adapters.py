"""Adapters only see a label-free request. Secrets are read from the environment."""

import json
import os
import selectors
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .core import digest, read_jsonl, request_digest


class AdapterError(RuntimeError):
    def __init__(self, code, status=None):
        super().__init__(code)
        self.code = code
        self.status = status


class BaseAdapter:
    def close(self):
        pass


class Baseline(BaseAdapter):
    def __init__(self, config):
        self.config = config
        self.metadata = {"adapter": "constant-baseline", "choice": config.get("choice")}

    def predict(self, request):
        options = list(request["questions"]["judge"]["criteria"])
        choice = self.config.get("choice", options[0])
        if choice not in options:
            raise AdapterError("baseline_choice_not_in_options")
        return {"choice": choice, "probabilities": {k: float(k == choice) for k in options}}, {}


class Replay(BaseAdapter):
    def __init__(self, config):
        path = Path(config["records"])
        rows = read_jsonl(path)
        self.rows = {r["case_id"]: r for r in rows if r.get("status") == "ok"}
        self.metadata = {
            "adapter": "replay",
            "records_sha256": digest(rows),
            "note": "Offline replay is not a new model measurement",
        }

    def predict(self, request):
        row = self.rows.get(request["id"])
        if row is None or row["request_hash"] != request_digest(request):
            raise AdapterError("replay_request_mismatch")
        return row["decision"], {"replay": True}


class TypeSafe(BaseAdapter):
    def __init__(self, config):
        self.config = config
        self.key = os.environ.get(config.get("key_env", "TYPESAFE_API_KEY"))
        if not self.key:
            raise AdapterError("missing_typesafe_key")
        self.metadata = {
            "adapter": "typesafe-api",
            "model_requested": config["model"],
            "endpoint": "https://api.typesafe.ai/v1/systemone",
        }

    def predict(self, request):
        payload = {
            "model": self.config["model"],
            "state": request["state"],
            "questions": request["questions"],
        }
        req = urllib.request.Request(
            self.metadata["endpoint"],
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
            method="POST",
        )

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        try:
            with urllib.request.build_opener(NoRedirect).open(
                req, timeout=self.config.get("timeout", 45)
            ) as res:
                result = json.load(res)
        except urllib.error.HTTPError as exc:
            raise AdapterError("http_error", exc.code) from None
        except (urllib.error.URLError, TimeoutError):
            raise AdapterError("transport_error") from None
        return result["answers"]["judge"], {
            "usage": result.get("usage"),
            "model_returned": result.get("model"),
        }


class Gateway(BaseAdapter):
    def __init__(self, config):
        self.timeout = config.get("timeout", 60)
        if not os.environ.get("AI_GATEWAY_API_KEY"):
            raise AdapterError("missing_gateway_key")
        self.proc = subprocess.Popen(
            [config.get("node", "node"), config["bridge"]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.model = config.get("model", "typesafe-ai/jev")
        self.metadata = {
            "adapter": "vercel-gateway",
            "model_requested": self.model,
            "bridge_sha256": digest(Path(config["bridge"]).read_text()),
            "backend_revision": "Use returned metadata; aliases are not pinned versions",
        }

    def predict(self, request):
        try:
            self.proc.stdin.write(json.dumps({**request, "model": self.model}) + "\n")
            self.proc.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.proc.stdout, selectors.EVENT_READ)
                if not selector.select(self.timeout):
                    self.proc.kill()
                    raise AdapterError("gateway_bridge_timeout")
            line = self.proc.stdout.readline()
            if not line:
                raise AdapterError("gateway_bridge_closed")
            result = json.loads(line)
        except (BrokenPipeError, json.JSONDecodeError):
            raise AdapterError("gateway_bridge_protocol_error") from None
        if not result.get("ok"):
            raise AdapterError(result.get("error", "gateway_error"), result.get("status"))
        return result["decision"], result.get("metadata", {})

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


class Laya(BaseAdapter):
    def __init__(self, config):
        import laya
        import torch
        from huggingface_hub import snapshot_download
        from laya.common import build_sequence, render_options

        self.build_sequence = build_sequence
        self.render_options = render_options
        torch.set_num_threads(config.get("threads", 4))
        revision = config.get("revision")
        if not revision:
            raise AdapterError("laya_revision_required")
        sub = config.get("subfolder", "")
        patterns = (
            [f"{sub}/*"] if sub else ["encoder/*", "tokenizer/*", "rl_agent_config.json", "model.safetensors"]
        )
        path = Path(
            snapshot_download(
                config.get("model", "convaiinnovations/laya"), revision=revision, allow_patterns=patterns
            )
        )
        self.agent = laya.load(str(path / sub), device=config.get("device", "cpu"))
        if config.get("max_len"):
            self.agent.cfg["max_len"] = config["max_len"]
        self.metadata = {
            "adapter": "laya",
            "model_requested": config.get("model", "convaiinnovations/laya"),
            "revision_requested": revision,
            "revision_resolved": path.name,
            "subfolder": sub,
            "device": str(self.agent.device),
            "torch": torch.__version__,
            "config": self.agent.cfg,
            "temperature": self.agent.temperature,
            "temperature_by_options": self.agent.temperature_by_options,
        }

    def predict(self, request):
        question = request["questions"]["judge"]
        internal = self.agent._to_internal(question)
        head = self.agent.cfg.get("head_max_len", 192)
        for text in self.render_options(internal):
            cleaned = text.replace(self.agent.tok.mask_token, " ")
            if len(self.agent.tok(" " + cleaned, add_special_tokens=False)["input_ids"]) > 48:
                raise AdapterError("laya_option_exceeds_sdk_token_cap")
        full, _ = self.build_sequence(self.agent.tok, request["state"], internal, 32768, head)
        generous, _ = self.build_sequence(self.agent.tok, request["state"], internal, 32768, 16384)
        if full != generous:
            raise AdapterError("laya_instruction_or_option_truncated")
        if len(full) > self.agent.cfg.get("max_len", 512):
            raise AdapterError("laya_state_exceeds_context")
        result = self.agent.predict(request["state"], request["questions"])
        if result["usage"]["input_tokens"] != len(full):
            raise AdapterError("laya_unexpected_truncation")
        return result["answers"]["judge"], {
            "usage": result["usage"],
            "device": str(self.agent.device),
            "input_tokens_verified": len(full),
        }


def create_adapter(config):
    name = config["adapter"]
    classes = {"baseline": Baseline, "replay": Replay, "typesafe": TypeSafe, "gateway": Gateway, "laya": Laya}
    if name not in classes:
        raise ValueError(f"Unknown adapter: {name}")
    return classes[name](config)
