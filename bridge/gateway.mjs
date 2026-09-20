import {createInterface} from 'node:readline';
import {experimental_evaluate as evaluate} from 'ai';
import {createGateway} from '@ai-sdk/gateway';
const gateway=createGateway({apiKey:process.env.AI_GATEWAY_API_KEY,fetch:async(url,init)=>{
  if(new URL(String(url)).hostname!=='ai-gateway.vercel.sh')throw new Error('unexpected_host');
  return fetch(url,{...init,redirect:'error'});
}});
for await(const line of createInterface({input:process.stdin,crlfDelay:Infinity})){
  try{
    const req=JSON.parse(line);
    const r=await evaluate({model:gateway.evaluationModel(req.model),state:req.state,questions:req.questions,
      maxRetries:0,abortSignal:AbortSignal.timeout(45000)});
    process.stdout.write(JSON.stringify({ok:true,decision:r.answers.judge,metadata:{usage:r.usage,
      model_returned:r.response.modelId,rounding:r.rounding,
      vendor_confidence:r.providerMetadata?.typesafe?.confidence?.judge??null}})+'\n');
  }catch(e){
    // Do not persist headers, credentials, provider response bodies, or arbitrary exception text.
    process.stdout.write(JSON.stringify({ok:false,error:'gateway_request_failed',status:e.statusCode??null})+'\n');
  }
}
