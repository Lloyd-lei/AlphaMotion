import {test} from 'node:test';
import assert from 'node:assert/strict';
import {poseElements} from './geometryMath.js';
function multiply(a,b){const r=Array(16).fill(0);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)r[col*4+row]+=a[k*4+row]*b[col*4+k];return r;}
test('nested MJCF object local rotation composes to a finite world transform',()=>{
 const sourceDescriptor={local_pos:[1,2,3],local_rot:[[1,0,0],[0,1,0],[0,0,1]]};
 const parent=poseElements([4,5,6],new Float32Array([0,-1,0,1,0,0,0,0,1]));
 const local=poseElements(sourceDescriptor.local_pos,sourceDescriptor.local_rot);const world=multiply(parent,local);
 assert.ok(world.every(Number.isFinite));assert.deepEqual(world.slice(12,15),[2,6,9]);
 assert.deepEqual([world[0],world[1],world[4],world[5]],[0,1,-1,0]);
});
test('flat and nested object descriptors produce identical WebGL matrix elements',()=>{
 const nested=[[0,-1,0],[1,0,0],[0,0,1]];assert.deepEqual(poseElements([.12,.34,.56],nested),poseElements(new Float32Array([.12,.34,.56]),new Float32Array(nested.flat())).map((x,i)=>i>=12&&i<=14?[.12,.34,.56][i-12]:x));
});
test('nonfinite rotation rejects rendering rather than silently hiding the object',()=>assert.throws(()=>poseElements([0,0,0],[[1,0,0],[0,NaN,0],[0,0,1]]),/非有限/));
test('full geom arrays can be indexed without interpreting neighbouring poses',()=>assert.deepEqual(poseElements([100,200,300,1,2,3],[1,0,0,0,1,0,0,0,1,0,-1,0,1,0,0,0,0,1],3,9).slice(12,15),[1,2,3]));
test('actual ScoccerBall_A_1 descriptor is finite and has correct composed translation',async()=>{
 const fs=await import('node:fs');const evidence=JSON.parse(fs.readFileSync(new URL('../.qa/source-object-descriptor.json',import.meta.url),'utf8'));
 assert.equal(evidence.object_name,'ScoccerBall_A_1');assert.equal(evidence.sha256,'5c0a01df2ce80aafa7060efe0913e567ac6905974576925668de0773906a92e8');
 for(const d of evidence.object_geoms){assert.ok(Array.isArray(d.local_rot[0]));const local=poseElements(d.local_pos,d.local_rot),parent=poseElements([4,5,6],[[0,-1,0],[1,0,0],[0,0,1]]),world=multiply(parent,local);assert.ok(world.every(Number.isFinite));assert.deepEqual(world.slice(12,15),[4-d.local_pos[1],5+d.local_pos[0],6+d.local_pos[2]]);}
});
