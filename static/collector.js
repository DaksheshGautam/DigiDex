const $=id=>document.getElementById(id);
const labels=JSON.parse($('supported-labels').textContent);
const canvas=$('drawing'),ctx=canvas.getContext('2d');
canvas.width=Math.max(280,Math.min(1000,Math.round(canvas.parentElement.getBoundingClientRect().width)));
let strokes=[],active=null,drawingId=crypto.randomUUID();
function redraw(){ctx.fillStyle='black';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='white';ctx.lineCap='round';ctx.lineJoin='round';for(const stroke of strokes){ctx.lineWidth=stroke.width;ctx.beginPath();ctx.moveTo(...stroke.points[0]);if(stroke.points.length===1)ctx.lineTo(stroke.points[0][0]+.1,stroke.points[0][1]+.1);else for(const point of stroke.points.slice(1))ctx.lineTo(...point);ctx.stroke()}}
function point(event){const box=canvas.getBoundingClientRect();return[(event.clientX-box.left)*canvas.width/box.width,(event.clientY-box.top)*canvas.height/box.height]}
canvas.addEventListener('pointerdown',event=>{if(strokes.length>=512){$('save-status').textContent='Save or clear this drawing before adding more strokes.';return}canvas.setPointerCapture(event.pointerId);active={width:Number($('pen').value),points:[point(event)]};strokes.push(active);redraw()});
canvas.addEventListener('pointermove',event=>{if(active){active.points.push(point(event));redraw()}});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>active=null);
function clear(){strokes=[];active=null;drawingId=crypto.randomUUID();redraw()}
$('clear').onclick=clear;$('undo').onclick=()=>{strokes.pop();active=null;redraw()};
function writer(){const value=$('writer').value;if(!/^[A-Za-z0-9_.:-]{1,64}$/.test(value))throw Error('Use 1–64 letters, numbers, dots, dashes or underscores for the writer ID.');return value}
try{$('writer').value=localStorage.getItem('glyphdex-private-writer')||'owner'}catch{}
$('writer').onchange=()=>{try{localStorage.setItem('glyphdex-private-writer',writer());$('writer').setCustomValidity('')}catch(error){$('writer').setCustomValidity(error.message);$('writer').reportValidity()}};
$('writer').oninput=()=>$('writer').setCustomValidity('');
const sync=new DatasetSync({storage:{getItem:key=>localStorage.getItem(key),setItem:(key,value)=>localStorage.setItem(key,value)},fetcher:(...args)=>fetch(...args),onState:state=>{
 $('save').disabled=!state.enabled;$('auto').disabled=!state.authenticated;$('auto').checked=state.enabled;$('disconnect').disabled=!state.authenticated;$('connect').disabled=!state.configured;
 $('dataset-status').textContent=state.error||(!state.configured?'Private storage is unavailable.':state.authenticated?`${state.characters} character${state.characters===1?'':'s'} and ${state.lines} line${state.lines===1?'':'s'} saved online · ${state.pending} queued · Sync ${state.enabled?'on':'paused'}.`:`Connect to save privately. ${state.pending} drawings queued in this browser.`);
 if(state.enabled&&$('save-status').textContent==='Connect the private dataset before saving.')$('save-status').textContent='Ready. Choose a label and draw it.';
}});
function packedStrokes(){const budget=Math.max(2,Math.min(4998,Math.floor(19000/Math.max(1,strokes.length))));return strokes.map(stroke=>({width:stroke.width,points:stroke.points.filter((point,i)=>i===0||i===stroke.points.length-1||i%Math.max(1,Math.ceil(stroke.points.length/budget))===0).map(([x,y])=>[Math.max(-40,Math.min(canvas.width+40,x)),Math.max(-40,Math.min(canvas.height+40,y))])}))}
function modeFor(text){const characters=Array.from(text).filter(c=>c!==' ');const groups={digits:labels.slice(0,10),uppercase:labels.slice(10,36),lowercase:labels.slice(36,62),symbols:labels.slice(62)};return Object.keys(groups).find(group=>characters.every(c=>groups[group].includes(c)))||'mixed'}
$('save').onclick=()=>{try{
 if(!sync.enabled)throw Error('Connect and enable syncing before saving.');
 const text=$('target').value,kind=$('kind').value,characters=Array.from(text);
 if(!text||text.trim()!==text||characters.some(c=>c!==' '&&!labels.includes(c)))throw Error('Enter exactly what you drew, using the supported labels.');
 if(kind==='glyph'&&!labels.includes(text))throw Error('For one character, enter exactly one supported letter, digit or symbol.');
 if(kind==='line'&&(characters.filter(c=>c!==' ').length>32||text.length>64))throw Error('Keep the drawing to one line and up to 32 characters.');
 if(!strokes.length||active)throw Error('Finish a drawing before saving.');
 const ink=ctx.getImageData(0,0,canvas.width,canvas.height).data;let pixels=0;for(let i=0;i<ink.length;i+=4)if(ink[i]>25)pixels++;if(pixels<3)throw Error('Draw a clear example before saving.');
 const id=writer(),sample={client_id:`${id}:${drawingId}:${kind}`,kind,drawing_id:drawingId,writer_group:id,mode:modeFor(text),image:canvas.toDataURL('image/png'),strokes:packedStrokes()};sample[kind==='glyph'?'label':'text']=text;
 if(!sync.capture(sample))throw Error(sync.state().error||'The drawing could not be queued.');
 $('save-status').textContent=`“${text}” queued privately. Check the online count above for sync confirmation. Draw another example or change the label.`;clear();
 }catch(error){$('save-status').textContent=error.message}};
function updateKind(){const single=$('kind').value==='glyph';$('sample-hint').textContent=single?'One plain character, including all its strokes.':'A word, number or short expression. Keep printed letters separated.';$('target').placeholder=single?'For example: A':'For example: Hello or A+7'}
$('kind').onchange=updateKind;
$('suggest').onclick=()=>{const category=$('category').value;const groups={mixed:labels,digits:labels.slice(0,10),uppercase:labels.slice(10,36),lowercase:labels.slice(36,62),symbols:labels.slice(62)};const pool=groups[category];let text;if($('kind').value==='glyph')text=pool[Math.floor(Math.random()*pool.length)];else if(category==='mixed')text=['Hello','World','A+7','Day 2','Good','Test','Code','Book','aB9','Hi 5'][Math.floor(Math.random()*10)];else text=Array.from({length:4},()=>pool[Math.floor(Math.random()*pool.length)]).join('');$('target').value=text};
$('target').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('save').click()}});
$('connect').onclick=async()=>{const key=$('key').value;$('key').value='';$('connect').disabled=true;try{await sync.connect(key)}catch(error){$('dataset-status').textContent=error.message}finally{$('connect').disabled=!sync.configured}};
$('auto').onchange=()=>sync.setEnabled($('auto').checked);
$('sync').onclick=async()=>{await sync.refresh();await sync.flush()};
$('disconnect').onclick=async()=>{try{await sync.disconnect()}catch(error){$('dataset-status').textContent=error.message}};
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$('export').onclick=async()=>{try{download(await sync.exportAll(),'glyphdex-private-dataset.json')}catch(error){$('dataset-status').textContent=error.message}};
$('export-queue').onclick=()=>download(sync.exportPending(),'glyphdex-private-queue.json');
window.addEventListener('online',()=>sync.flush());sync.refresh();redraw();
