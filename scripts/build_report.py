"""Build a portable offline comparison page from a task and named result logs."""

import argparse
import html
import json
from pathlib import Path

from decision_model_bench.core import digest, load_task, read_jsonl, request_digest, request_for
from decision_model_bench.metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--run", action="append", required=True, help="name=attempts.jsonl; repeat for each model"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    task, cases, _ = load_task(args.task)
    models = {}
    data = []
    summaries = {}
    hashes = {c["id"]: request_digest(request_for(c, task)) for c in cases}
    for run in args.run:
        name, path = run.split("=", 1)
        if name in models:
            raise ValueError("Duplicate model name")
        rows = read_jsonl(path)
        if any(r.get("request_hash") != hashes.get(r["case_id"]) for r in rows):
            raise ValueError("Mismatched task requests")
        case_hashes = {c["id"]: digest(c) for c in cases}
        if any(r.get("case_hash") != case_hashes.get(r["case_id"]) for r in rows):
            raise ValueError("Mismatched case targets or metadata")
        summaries[name] = summarize(task, cases, rows)
        models[name] = {r["case_id"]: r for r in rows}
    groups = task.get("label_groups", {k: k for k in task["question"]["criteria"]})
    for case in cases:
        entry = {**case, "models": {}}
        for name, rows in models.items():
            row = rows.get(case["id"])
            if row and row["status"] == "ok":
                entry["models"][name] = {
                    "decision": row["decision"],
                    "correct": groups[row["decision"]["choice"]] == groups[case["target"]],
                }
            else:
                entry["models"][name] = {"error": row["status"] if row else "unattempted"}
        data.append(entry)
    content = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Decision Model Bench</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 20px;background:#11151d;color:#eee}p{line-height:1.7;color:#aab6c9}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #343f51}button,input,select{font:inherit;margin:8px 8px 8px 0;padding:10px;border-radius:8px;background:#1c2532;color:#eee;border:1px solid #596779}pre{white-space:pre-wrap;line-height:1.6}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{background:#1c2532;padding:20px;border-radius:10px;flex:1;min-width:200px}.ok{color:#75d9ed}.bad{color:#f3c775}</style><h1>Decision Model Bench</h1><p>TASK</p><p>Accuracy includes failed and unattempted cases as wrong. Check coverage before comparing. Label grouping follows the task definition. This page does not imply a controlled speed comparison.</p><div id="summary"></div><h2>Cases</h2><select id="filter"><option value="all">All cases</option><option value="disagree">Different choices</option><option value="mixed">Mixed correctness</option></select><input id="search" placeholder="Search state or group"><button id="prev">Previous</button><button id="next">Next</button><p id="count"></p><pre id="state"></pre><p id="target"></p><div id="cards" class="cards"></div><script id="payload" type="application/json">PAYLOAD</script><script>
const {data,scores}=JSON.parse(document.getElementById('payload').textContent);const names=Object.keys(scores);let index=0,rows=data;const el=id=>document.getElementById(id);const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
el('summary').innerHTML='<table><tr><th>Model</th><th>Attempted</th><th>Accuracy</th><th>Failed attempts</th><th>Event Brier</th></tr>'+names.map(n=>{const s=scores[n];return `<tr><td>${esc(n)}</td><td>${s.attempted}/${s.total}</td><td>${(s.accuracy*100).toFixed(2)}%</td><td>${s.failed_attempts}</td><td>${s.event_brier===undefined?'—':s.event_brier?.toFixed(4)??'—'}</td></tr>`}).join('')+'</table>';
function draw(){el('count').textContent=rows.length?`${index+1}/${rows.length} · ID ${rows[index].id}`:'No matching cases';if(!rows.length){el('state').textContent='';el('target').textContent='';el('cards').innerHTML='';return}const r=rows[index];el('state').textContent=r.state;el('target').textContent=`Target: ${r.target} · Group: ${r.group??r.id}`;el('cards').innerHTML=names.map(n=>{const a=r.models[n];return `<div class="card"><b>${esc(n)}</b>${a.error?`<p>${esc(a.error)}</p>`:`<h3 class="${a.correct?'ok':'bad'}">${esc(a.decision.choice)} · ${a.correct?'Correct':'Wrong'}</h3><pre>${esc(JSON.stringify(a.decision.probabilities,null,2))}</pre>`}</div>`}).join('')}
function apply(){const q=el('search').value.toLowerCase(),f=el('filter').value;rows=data.filter(r=>(r.state+' '+(r.group??'')).toLowerCase().includes(q)&&(f==='all'||f==='disagree'&&new Set(names.map(n=>r.models[n].decision?.choice??r.models[n].error)).size>1||f==='mixed'&&new Set(names.map(n=>r.models[n].correct??false)).size>1));index=0;draw()}el('filter').onchange=apply;el('search').oninput=apply;el('next').onclick=()=>{if(rows.length){index=(index+1)%rows.length;draw()}};el('prev').onclick=()=>{if(rows.length){index=(index+rows.length-1)%rows.length;draw()}};draw();</script></html>"""
    content = content.replace(">TASK<", ">" + html.escape(task["id"]) + "<")
    content = content.replace(
        ">PAYLOAD<",
        ">"
        + json.dumps({"data": data, "scores": summaries}, ensure_ascii=False).replace("<", "\\u003c")
        + "<",
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(content, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
