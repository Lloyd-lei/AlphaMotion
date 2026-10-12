// Row-major source rotations -> column-major WebGL transform. No position quantisation.
export function poseElements(position,rotation,offsetP=0,offsetR=0){
 const r=Array.isArray(rotation?.[0])?rotation.flat():rotation;
 if(!position||!r||position.length<offsetP+3||r.length<offsetR+9)throw new Error('几何位姿维度不正确');
 const v=[r[offsetR],r[offsetR+3],r[offsetR+6],0,r[offsetR+1],r[offsetR+4],r[offsetR+7],0,r[offsetR+2],r[offsetR+5],r[offsetR+8],0,position[offsetP],position[offsetP+1],position[offsetP+2],1];
 if(!v.every(Number.isFinite))throw new Error('几何位姿包含非有限数值');
 return v;
}
