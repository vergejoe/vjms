'use strict';
const $ = id => document.getElementById(id);
const state = {system:null,datasets:[],tokenizers:[],runs:[],run:null,metrics:[],output:'',ready:false};
const titles = {overview:'概览',data:'数据集',tokenizer:'分词器',training:'模型与训练',playground:'生成实验室',deployment:'本机部署'};
const statuses = {queued:'准备中',running:'训练中',paused:'已暂停',pausing:'暂停中',completed:'已完成',failed:'失败',interrupted:'已中断'};
const active = status => ['queued','running','pausing'].includes(status);
const esc = value => String(value ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = value => Number(value || 0).toLocaleString('en-US');
const fixed = (value,digits=3) => value == null ? '—' : Number(value).toFixed(digits);
let toastTimer;
function toast(message){$('toast').textContent=message;$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,6000);}
async function api(path,options={}){
  const response=await fetch(path,{...options,headers:options.body instanceof FormData?{}:{'Content-Type':'application/json',...options.headers}});
  const data=await response.json();
  if(!response.ok){let detail=data.detail;if(Array.isArray(detail))detail=detail.map(x=>`${x.loc?.slice(1).join('.')}: ${x.msg}`).join('\n');throw new Error(detail||`请求失败 (${response.status})`);}
  return data;
}
const post=(path,data)=>api(path,{method:'POST',body:JSON.stringify(data)});
async function busy(form,action,message){
  const button=form.querySelector('[type="submit"]');
  if(button?.disabled)return;
  const old=button?.textContent;
  if(button){button.disabled=true;button.textContent=message||'正在处理…';}
  try{await action();}catch(error){toast(error.message);}finally{if(button){button.disabled=false;button.textContent=old;}}
}
function navigate(){
  const page=location.hash.slice(1) in titles?location.hash.slice(1):'overview';
  document.querySelectorAll('.page').forEach(el=>el.hidden=el.id!==`page-${page}`);
  document.querySelectorAll('nav a').forEach(el=>{const selected=el.dataset.page===page;el.classList.toggle('active',selected);if(selected)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  document.body.dataset.page=page;
  document.title=`${titles[page]} · vjms`;
  $('breadcrumb').textContent=titles[page];
  window.scrollTo(0,0);
  if(page==='training')drawChart();
}
document.querySelector('.skip-link').addEventListener('click',event=>{
  event.preventDefault();$('main-content').focus();$('main-content').scrollIntoView();
});
window.addEventListener('hashchange',navigate);
navigate();
function options(id,items,label,empty){
  const element=$(id),previous=element.value;
  const next=items.length?items.map(item=>`<option value="${esc(item.id)}">${esc(label(item))}</option>`).join(''):`<option value="">${esc(empty)}</option>`;
  if(element.innerHTML!==next){element.innerHTML=next;if(items.some(x=>x.id===previous))element.value=previous;}
}
function readyRuns(){return state.runs.filter(x=>x.step>0);}
function renderLists(){
  $('stat-datasets').textContent=state.datasets.length;
  $('stat-tokenizers').textContent=state.tokenizers.length;
  $('stat-runs').textContent=state.runs.length;
  const running=state.runs.filter(x=>active(x.status)).length;
  $('stat-runs-note').textContent=running?`${running} 个实验正在运行`:state.runs.length?'已保存的训练实验':'尚未开始训练';
  $('dataset-count').textContent=`${state.datasets.length} DATASETS`;
  $('dataset-list').innerHTML=state.datasets.map(x=>`<article class="resource-card"><h3>▤ ${esc(x.name)}</h3><div class="resource-meta"><span>${num(x.characters)} 字符</span><span>${num(x.train_documents)} 训练 / ${num(x.validation_documents)} 验证</span></div><details><summary>查看文本预览</summary><div class="resource-preview">${esc(x.preview)}</div></details></article>`).join('')||'<div class="empty-card">还没有数据集。导入文本，或先载入中英文示例。</div>';
  $('tokenizer-list').innerHTML=state.tokenizers.map(x=>`<article class="resource-card"><h3>⌗ ${esc(x.dataset_name)}</h3><div class="resource-meta"><span>${num(x.vocab_size)} 词表</span><span>Byte-level BPE</span></div><p class="helper">训练 ${num(x.train_tokens)} tokens<br>验证 ${num(x.validation_tokens)} tokens<br>仅在训练集上学习词表</p></article>`).join('')||'<div class="empty-card">选择数据集，训练你的第一个分词器。</div>';
  options('tokenizer-dataset',state.datasets,x=>x.name,'请先导入数据集');
  for(const id of ['encode-tokenizer','run-tokenizer'])options(id,state.tokenizers,x=>`${x.dataset_name} · ${num(x.vocab_size)} 词`,'请先训练分词器');
  options('monitor-run',state.runs,x=>`${x.name} · ${statuses[x.status]||x.status}`,'尚无训练实验');
  for(const id of ['generate-run','deploy-run'])options(id,readyRuns(),x=>`${x.name} · ${(x.parameters/1e6).toFixed(2)}M`,'请先训练并保存检查点');
  const runHTML=(x,button=true)=>`<div class="run-item"><span class="badge">${esc(statuses[x.status]||x.status)}</span><div><b>${esc(x.name)}</b><p>${esc(x.dataset_name)} · ${(x.parameters/1e6).toFixed(2)}M 参数</p></div><span class="mono">${num(x.step)} / ${num(x.max_steps)}</span>${button?`<button class="button subtle" data-observe="${esc(x.id)}">查看 →</button>`:''}</div>`;
  $('run-list').innerHTML=state.runs.map(x=>runHTML(x)).join('')||'<div class="empty">还没有实验。完成左侧配置，开始从零训练。</div>';
  if(state.runs.length)$('recent-runs').innerHTML=state.runs.slice(0,3).map(x=>runHTML(x)).join('');
  $('recent-runs').classList.toggle('empty',!state.runs.length);
  updateApiExample();
}
function getRunRequest(){
  return {name:$('run-name').value,tokenizer_id:$('run-tokenizer').value,
    architecture:{dim:+$('dim').value,layers:+$('layers').value,heads:+$('heads').value,hidden_dim:+$('hidden-dim').value,context_length:+$('context-length').value},
    device:$('device').value,max_steps:+$('max-steps').value,batch_size:+$('batch-size').value,
    gradient_accumulation:+$('accumulation').value,learning_rate:+$('learning-rate').value,
    eval_interval:+$('eval-interval').value,checkpoint_interval:+$('checkpoint-interval').value};
}
let estimateTicket=0;
async function estimate(){
  const ticket=++estimateTicket;
  if(!$('run-tokenizer').value){$('model-estimate').textContent='选择一个分词器，计算实际参数量。';return;}
  try{const result=await post('/api/models/estimate',getRunRequest());if(ticket!==estimateTicket)return;
    $('model-estimate').classList.remove('invalid');
    $('model-estimate').innerHTML=`<strong>${num(result.parameters)}</strong> 参数 · FP32 权重 <strong>${fixed(result.weight_mb_fp32,1)} MB</strong><br>基础训练状态 ${fixed(result.training_state_mb_fp32,1)} MB，不含激活与框架开销。`;
  }catch(error){if(ticket!==estimateTicket)return;$('model-estimate').classList.add('invalid');$('model-estimate').textContent=error.message;}
}
let estimateTimer;
$('run-form').addEventListener('input',()=>{clearTimeout(estimateTimer);estimateTimer=setTimeout(estimate,200);});
$('run-tokenizer').addEventListener('change',estimate);
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{
  if(!state.system)return;
  const preset=state.system.presets[button.dataset.preset];
  for(const [key,id] of Object.entries({dim:'dim',layers:'layers',heads:'heads',hidden_dim:'hidden-dim'}))$(id).value=preset[key];
  document.querySelectorAll('[data-preset]').forEach(x=>{x.classList.toggle('selected',x===button);x.setAttribute('aria-pressed',String(x===button));});estimate();
}));
async function refresh(){
  [state.datasets,state.tokenizers,state.runs]=await Promise.all([api('/api/datasets'),api('/api/tokenizers'),api('/api/runs')]);
  renderLists();await refreshMonitor();
}
async function addSample(){const data=await post('/api/datasets/sample',{});await refresh();$('tokenizer-dataset').value=data.id;toast('中英文流程示例已导入。下一步：训练分词器。');location.hash='data';}
for(const id of ['overview-sample','sample-button'])$(id).addEventListener('click',async()=>{const button=$(id);button.disabled=true;try{await addSample();}catch(error){toast(error.message);}finally{button.disabled=false;}});
$('dataset-form').addEventListener('submit',event=>{event.preventDefault();busy(event.currentTarget,async()=>{
  const data=await post('/api/datasets',{name:$('dataset-name').value,text:$('dataset-text').value,validation_fraction:+$('validation-fraction').value});
  await refresh();$('tokenizer-dataset').value=data.id;$('dataset-text').value='';toast('数据集已保存，训练集和验证集已划分。');
});});
$('file-input').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file)return;
  if(file.size>32*1024*1024){toast('文件超过 32MB 上限。');event.target.value='';return;}
  event.target.disabled=true;
  try{const body=new FormData();body.append('file',file);body.append('validation_fraction',$('validation-fraction').value);
    const data=await api('/api/datasets/upload',{method:'POST',body});await refresh();$('tokenizer-dataset').value=data.id;toast('文本文件已导入。');
  }catch(error){toast(error.message);}finally{event.target.disabled=false;event.target.value='';}
});
$('tokenizer-form').addEventListener('submit',event=>{event.preventDefault();busy(event.currentTarget,async()=>{
  const result=await post('/api/tokenizers',{dataset_id:$('tokenizer-dataset').value,vocab_size:+$('vocab-size').value});
  await refresh();$('encode-tokenizer').value=result.id;$('run-tokenizer').value=result.id;await estimate();toast(`分词器已保存：实际词表 ${num(result.vocab_size)}，可前往模型与训练。`);
},'正在学习词表…');});
$('encode-form').addEventListener('submit',event=>{event.preventDefault();busy(event.currentTarget,async()=>{
  const result=await post(`/api/tokenizers/${$('encode-tokenizer').value}/encode`,{text:$('encode-text').value});
  $('encode-result').innerHTML=result.tokens.slice(0,300).map((token,i)=>`<span class="token-chip" title="ID ${result.ids[i]}">${esc(token)}</span>`).join('')+`<p class="helper">${result.ids.length} tokens · 解码：${esc(result.decoded)}${result.tokens.length>300?' · 仅展示前 300 个 Token':''}</p>`;
});});
$('run-form').addEventListener('submit',event=>{event.preventDefault();busy(event.currentTarget,async()=>{
  const result=await post('/api/runs',getRunRequest());await refresh();$('monitor-run').value=result.id;await refreshMonitor();toast('训练进程已启动。可随时保存并暂停。');
},'正在启动训练…');});
document.addEventListener('click',event=>{const button=event.target.closest('[data-observe]');if(!button)return;$('monitor-run').value=button.dataset.observe;location.hash='training';refreshMonitor().catch(error=>toast(error.message));});
$('monitor-run').addEventListener('change',()=>refreshMonitor().catch(error=>toast(error.message)));
async function refreshMonitor(){
  const id=$('monitor-run').value;if(!id)return;
  const run=await api(`/api/runs/${id}`);if($('monitor-run').value!==id)return;
  state.run=run;state.metrics=run.metrics;
  $('run-status').textContent=statuses[run.status]||run.status;
  const last=run.metrics.at(-1),val=[...run.metrics].reverse().find(x=>x.val_loss!=null);
  $('metric-loss').textContent=fixed(last?.train_loss);$('metric-val').textContent=fixed(val?.val_loss);$('metric-speed').textContent=last?num(Math.round(last.tokens_per_second)):'—';
  $('progress-label').textContent=`${num(run.step)} / ${num(run.max_steps)} STEPS`;
  $('run-progress').max=run.max_steps;$('run-progress').value=run.step;
  $('chart-empty').hidden=run.metrics.length>0;
  $('pause-button').disabled=!active(run.status);
  $('resume-button').disabled=active(run.status)||!run.checkpoints.includes('latest');
  $('run-error').hidden=!run.error;$('run-error').textContent=run.error||'';
  const seconds=run.elapsed_seconds||0;
  $('run-facts').textContent=`${(run.parameters/1e6).toFixed(2)}M 参数 · ${run.device||run.config.device} · ${run.precision||'等待启动'} · 累计 ${Math.round(seconds)} 秒 · ${num(run.tokens_seen)} tokens`;
  $('checkpoint-links').innerHTML=run.checkpoints.map(name=>`<a href="/api/runs/${run.id}/checkpoints/${name}" download>↓ ${name} 检查点</a>`).join('');
  $('metrics-body').innerHTML=run.metrics.slice(-8).reverse().map(x=>`<tr><td>${num(x.step)}</td><td>${fixed(x.train_loss,4)}</td><td>${fixed(x.val_loss,4)}</td><td>${num(Math.round(x.tokens_per_second))}</td></tr>`).join('')||'<tr><td colspan="4" class="table-empty">等待第一个训练步</td></tr>';
  drawChart();
}
for(const action of ['pause','resume'])$(`${action}-button`).addEventListener('click',async()=>{
  const id=$('monitor-run').value;if(!id)return;
  const button=$(`${action}-button`);button.disabled=true;
  try{const additional=+$('resume-steps').value;if(action==='resume'&&(!Number.isInteger(additional)||additional<1||additional>1000000))throw new Error('追加步数需要是 1–1,000,000 的整数。');
    await post(`/api/runs/${id}/${action}`,action==='resume'?{additional_steps:additional,device:$('device').value}:{});
    toast(action==='pause'?'已请求暂停，当前训练步结束后会保存。':'正在恢复模型、优化器和训练状态。');await refresh();
  }catch(error){toast(error.message);}finally{await refreshMonitor().catch(()=>{});}
});
function drawChart(){
  const canvas=$('loss-chart');if($('page-training').hidden)return;
  const {width,height}=canvas.getBoundingClientRect();if(!width)return;
  const ratio=devicePixelRatio||1;canvas.width=width*ratio;canvas.height=height*ratio;
  const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);ctx.clearRect(0,0,width,height);
  const left=38,right=12,top=15,bottom=26,w=width-left-right,h=height-top-bottom;
  const metrics=state.metrics,all=metrics.flatMap(x=>[x.train_loss,x.val_loss]).filter(x=>x!=null&&Number.isFinite(x));
  let low=all.length?Math.max(0,Math.min(...all)-.2):0,high=all.length?Math.max(...all)+.2:8;
  if(high-low<.5){low=Math.max(0,low-.25);high+=.25;}
  ctx.font='9px monospace';ctx.textAlign='right';
  for(let i=0;i<5;i++){const y=top+i*h/4;ctx.strokeStyle='#ededed';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(width-right,y);ctx.stroke();ctx.fillStyle='#858585';ctx.fillText((high-i*(high-low)/4).toFixed(1),left-9,y+3);}
  if(!metrics.length)return;
  const maxStep=Math.max(2,metrics.at(-1).step);
  const x=step=>left+(step-1)/(maxStep-1)*w,y=value=>top+(high-value)/(high-low)*h;
  for(const [key,color] of [['train_loss','#242424'],['val_loss','#6c8eac']]){
    const points=metrics.filter(m=>m[key]!=null);ctx.strokeStyle=color;ctx.lineWidth=1.8;ctx.lineJoin='round';ctx.beginPath();
    points.forEach((m,i)=>{if(i===0)ctx.moveTo(x(m.step),y(m[key]));else ctx.lineTo(x(m.step),y(m[key]));});ctx.stroke();
    if(key==='val_loss'||points.length===1){ctx.fillStyle=color;for(const m of points){ctx.beginPath();ctx.arc(x(m.step),y(m[key]),2.5,0,Math.PI*2);ctx.fill();}}
  }
  ctx.fillStyle='#858585';ctx.textAlign='left';ctx.fillText('1',left,height-7);ctx.textAlign='right';ctx.fillText(String(maxStep),width-right,height-7);
}
new ResizeObserver(drawChart).observe($('loss-chart'));
$('generate-form').addEventListener('submit',event=>{event.preventDefault();busy(event.currentTarget,async()=>{
  const result=await post('/api/generate',{run_id:$('generate-run').value,checkpoint:$('generate-checkpoint').value,prompt:$('prompt').value,max_new_tokens:+$('max-new-tokens').value,temperature:+$('temperature').value,top_k:+$('top-k').value,top_p:+$('top-p').value,repetition_penalty:+$('repetition').value});
  state.output=result.text;$('generation-output').replaceChildren();
  const prompt=document.createElement('span');prompt.className='prompt-text';prompt.textContent=result.prompt;
  const completion=document.createElement('span');completion.className='completion-text';completion.textContent=result.completion||'（模型生成了结束符）';
  $('generation-output').append(prompt,completion);$('copy-output').disabled=false;
  $('generation-stats').textContent=`${result.generated_tokens} TOKENS · ${result.seconds.toFixed(2)}s · ${result.tokens_per_second.toFixed(1)} TOKENS/s · ${result.device.toUpperCase()}${result.prompt_tokens>result.context_tokens?' · 提示词已截取末尾上下文':''}`;
},'正在生成…');});
function updateApiExample(){
  const id=$('deploy-run').value||'YOUR_RUN_ID';
  const payload=JSON.stringify({model:id,prompt:'从前有一只小猫',max_tokens:100,temperature:0.8,top_p:0.95},null,2);
  $('endpoint-url').textContent=`${location.origin}/v1/completions`;
  const slash=String.fromCharCode(92);
  $('api-example').textContent=[`curl ${location.origin}/v1/completions ${slash}`,`  -H "Content-Type: application/json" ${slash}`,`  -d '${payload}'`].join('\n');
}
$('deploy-run').addEventListener('change',updateApiExample);
async function copy(text){try{await navigator.clipboard.writeText(text);toast('已复制。');}catch{toast('浏览器未允许复制，请选中文本手动复制。');}}
$('copy-output').addEventListener('click',()=>copy(state.output));$('copy-api').addEventListener('click',()=>copy($('api-example').textContent));
async function boot(){
  try{state.system=await api('/api/system');const info=state.system;
    $('header-device').textContent=`${info.device.toUpperCase()} / ${info.ram_gb} GB`;$('stat-device').textContent=info.device.toUpperCase();
    $('stat-memory').textContent=`${info.gpu_name} · ${info.ram_gb} GB 内存`;$('data-path').textContent=info.data_dir;
    $('connection-text').textContent='本机服务已连接';await refresh();await estimate();state.ready=true;
  }catch(error){$('connection-text').textContent='连接失败';$('boot-error').hidden=false;$('connection-dot').style.background='#d27b60';}
}
let polling=false;
setInterval(async()=>{if(!state.ready||polling||document.hidden)return;polling=true;
  try{state.runs=await api('/api/runs');renderLists();await refreshMonitor();$('connection-text').textContent='本机服务已连接';}
  catch{$('connection-text').textContent='连接中断，请检查服务';}finally{polling=false;}
},3000);
boot();
