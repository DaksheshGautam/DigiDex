/* Durable correction queue. Credentials are never written to browser storage. */
(function(root){
class DatasetSync {
  constructor({storage,fetcher,onState=()=>{},autoFlush=true,makeRevision=()=>crypto.randomUUID()}){
    this.storage=storage;this.fetcher=fetcher;this.onState=onState;this.autoFlush=autoFlush;this.makeRevision=makeRevision;
    this.queueKey='glyphdex-private-queue-v1';this.preferenceKey='glyphdex-private-auto-v1';
    this.pending=new Map();this.authenticated=false;this.configured=false;this.enabled=false;this.csrf='';this.error='';this.counts={characters:0,lines:0};this.running=false;this.invalid=new Set();
    try{const saved=JSON.parse(storage.getItem(this.queueKey)||'[]');if(Array.isArray(saved))for(const item of saved){if(item?.sample?.client_id&&item.revision)this.pending.set(item.sample.client_id,item)}}catch{this.error='The local correction backup could not be read.'}
  }
  state(){return {authenticated:this.authenticated,configured:this.configured,enabled:this.enabled,pending:this.pending.size,characters:this.counts.characters,lines:this.counts.lines,error:this.error}}
  emit(){this.onState(this.state())}
  persist(){try{this.storage.setItem(this.queueKey,JSON.stringify([...this.pending.values()]));return true}catch{this.error='Browser storage is full or unavailable. Keep this tab open and export queued corrections before leaving.';return false}}
  async request(path,options={}){
    const response=await this.fetcher('/dataset/'+path,{credentials:'same-origin',...options});
    const data=await response.json();if(!response.ok){const error=new Error(data.error||'Private dataset unavailable.');error.status=response.status;throw error}return data;
  }
  async refresh(){try{const data=await this.request('status');this.configured=data.configured;this.authenticated=data.authenticated;this.csrf=data.csrf||'';if(data.authenticated){this.counts={characters:data.characters??this.counts.characters,lines:data.lines??this.counts.lines};this.enabled=this.storage.getItem(this.preferenceKey)!=='off'}else this.enabled=false;this.error=data.error||''}catch(e){this.error=e.message}this.emit();if(this.enabled)this.schedule()}
  async connect(key){const data=await this.request('session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});this.authenticated=data.authenticated;this.csrf=data.csrf;this.enabled=true;try{this.storage.setItem(this.preferenceKey,'on')}catch{}await this.refresh();this.schedule()}
  setEnabled(enabled){this.enabled=Boolean(enabled&&this.authenticated);try{this.storage.setItem(this.preferenceKey,this.enabled?'on':'off')}catch{}this.emit();if(this.enabled)this.schedule()}
  async disconnect(){await this.request('logout',{method:'POST',headers:{'X-Dataset-CSRF':this.csrf}});this.authenticated=false;this.enabled=false;this.csrf='';this.emit()}
  capture(sample){
    if(!this.enabled||!this.authenticated)return false;
    if(!sample?.client_id)throw new Error('A correction ID is required.');
    if(!this.pending.has(sample.client_id)&&this.pending.size>=5000){this.error='The local queue is full. Sync or export it before adding more.';this.emit();return false}
    this.pending.set(sample.client_id,{revision:this.makeRevision(),sample:JSON.parse(JSON.stringify(sample))});
    this.persist();this.emit();this.schedule();return true;
  }
  schedule(){if(!this.autoFlush)return;clearTimeout(this.timer);this.timer=setTimeout(()=>this.flush(),300)}
  async flush(){
    if(this.running||!this.enabled||!this.authenticated)return;
    this.running=true;let lastError='';
    try{while(this.enabled&&this.authenticated){
      const item=[...this.pending.values()].find(v=>!this.invalid.has(v.revision));if(!item)break;
      try{const data=await this.request('samples',{method:'POST',headers:{'Content-Type':'application/json','X-Dataset-CSRF':this.csrf},body:JSON.stringify({samples:[item.sample]})});
        // A late reply must never discard a newer correction for the same drawing.
        if(this.pending.get(item.sample.client_id)?.revision===item.revision)this.pending.delete(item.sample.client_id);
        this.counts={characters:data.characters,lines:data.lines};this.error='';this.persist();this.emit();
      }catch(e){lastError=e.message;this.error=e.message;if(e.status===401){this.authenticated=false;this.enabled=false;break}if(e.status===400){this.invalid.add(item.revision);this.emit();continue}break}
    }}finally{this.running=false;if(lastError)this.error=lastError;this.emit()}
  }
  async exportAll(){
    const output={schema_version:1,examples:[],lines:[]};let cursor='';
    do{const data=await this.request('export?limit=200'+(cursor?'&cursor='+encodeURIComponent(cursor):''));output.examples.push(...data.examples);output.lines.push(...data.lines);cursor=data.next_cursor}while(cursor);
    return output;
  }
  exportPending(){const output={schema_version:1,queued_only:true,examples:[],lines:[]};for(const {sample} of this.pending.values())output[sample.kind==='glyph'?'examples':'lines'].push(sample);return output}
}
if(typeof module!=='undefined'&&module.exports)module.exports=DatasetSync;else root.DatasetSync=DatasetSync;
})(typeof window!=='undefined'?window:globalThis);
