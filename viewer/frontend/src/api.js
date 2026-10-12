export class ApiError extends Error { constructor(message,status,detail={}){super(message);this.status=status;this.code=detail.code;this.committed=!!detail.committed;this.reload_required=!!detail.reload_required;} }
export async function request(path,options={}) {
 const r=await fetch(path,{...options,headers:{...(options.body?{'Content-Type':'application/json'}:{}),...options.headers}});
 if(!r.ok){let d;try{d=await r.json();}catch{d={detail:r.statusText};}throw new ApiError(typeof d.detail==='string'?d.detail:(d.detail?.message||JSON.stringify(d.detail||d)),r.status,d.detail||d);}
 return r;
}
export async function json(path,options){return (await request(path,options)).json();}
const CTORS={float32:Float32Array,uint32:Uint32Array,uint8:Uint8Array,int32:Int32Array,float64:Float64Array,int16:Int16Array,uint16:Uint16Array};
export function decode(buffer){
 const dv=new DataView(buffer);if(buffer.byteLength<4)throw new Error('二进制响应不完整');
 const n=dv.getUint32(0,true);if(n<2||n>buffer.byteLength-4)throw new Error('二进制响应头长度错误');
 const h=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,4,n)));if(h.schema!=='motion-audit-binary-v1')throw new Error('未知二进制协议版本：'+h.schema);const base=4+Math.ceil(n/4)*4, arrays={};
 for(const [key,d] of Object.entries(h.arrays||{})){const C=CTORS[d.dtype];if(!C)throw new Error('未知二进制类型 '+d.dtype);const len=d.shape.reduce((a,b)=>a*b,1),off=base+d.offset;if(off<base||off+len*C.BYTES_PER_ELEMENT>buffer.byteLength)throw new Error('二进制数组越界 '+key);let data;if(off%C.BYTES_PER_ELEMENT)data=new C(buffer.slice(off,off+len*C.BYTES_PER_ELEMENT));else data=new C(buffer,off,len);arrays[key]={data,shape:d.shape};}
 return {meta:h.meta||h,arrays,byteLength:buffer.byteLength};
}
export async function binary(path,signal){return decode(await (await request(path,{signal})).arrayBuffer());}
export const itemKey = item => item?.key||item?.item_key||item?.item_id;
export const itemPath = item => '/api/items/'+encodeURIComponent(itemKey(item));
export const values = (pack,name) => pack?.arrays?.[name]?.data;
export function row(pack,name,index){const a=pack?.arrays?.[name];if(!a)return null;const stride=a.shape.slice(1).reduce((x,y)=>x*y,1);return a.data.subarray(index*stride,(index+1)*stride);}
export const prettyRobot=s=>String(s||'').replace('unitree_','').replace('fourier_','').replace('_29dof',' · 29 DoF').replace('_23dof',' · 23 DoF').toUpperCase();
export const clipLabel=item=>item?.action||item?.action_category||item?.clip_id?.split('/').pop()||item?.item_id?.split('::')[0]||'未选择数据';
