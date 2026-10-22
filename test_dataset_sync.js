const assert=require('node:assert/strict');
const DatasetSync=require('./static/dataset-sync.js');
const memory=new Map();let revision=0;
const storage={getItem:k=>memory.get(k)||null,setItem:(k,v)=>memory.set(k,v)};
const reply=(data,status=200)=>({ok:status<400,status,json:async()=>data});
const glyph=label=>({client_id:'owner:draw:0',kind:'glyph',label});
(async()=>{
  let offline=true,saved=[];
  const fetcher=async(url,options)=>{
    if(url.endsWith('status'))return reply({configured:true,authenticated:true,csrf:'token',characters:saved.length,lines:0});
    if(url.endsWith('session'))return reply({authenticated:true,csrf:'token'});
    if(offline)throw Error('offline');
    saved.push(JSON.parse(options.body).samples[0]);return reply({characters:1,lines:0});
  };
  const make=()=>new DatasetSync({storage,fetcher,autoFlush:false,makeRevision:()=>String(++revision)});
  const sync=make();await sync.connect('a-key-that-must-never-be-saved');
  sync.capture(glyph('A'));await sync.flush();assert.equal(sync.state().pending,1);
  assert.ok(!JSON.stringify([...memory]).includes('a-key-that-must-never-be-saved'));
  const restored=make();await restored.refresh();restored.capture(glyph('B'));
  offline=false;await restored.flush();assert.equal(restored.state().pending,0);assert.equal(saved[0].label,'B');
  // A reply for an earlier label cannot erase a newer correction.
  let release;const gate=new Promise(resolve=>release=resolve);let calls=0;
  restored.fetcher=async()=>{if(++calls===1)await gate;return reply({characters:1,lines:0})};
  restored.capture(glyph('C'));const flushing=restored.flush();restored.capture(glyph('D'));release();await flushing;
  assert.equal(calls,2);assert.equal(restored.state().pending,0);
  restored.fetcher=async()=>reply({error:'expired'},401);restored.capture(glyph('E'));await restored.flush();
  assert.equal(restored.state().pending,1);assert.equal(restored.state().authenticated,false);
  assert.equal(restored.capture(glyph('F')),false);
  console.log('Dataset queue: offline persistence, relabeling, revision races, credential privacy and expired-session checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1});
