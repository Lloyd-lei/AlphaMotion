import {ProgressPage} from './ProgressPage';
import {GpuPage} from './GpuPage';
import {HumanPage} from './HumanPage';
import './atlas.css';
export const ATLAS_TABS=[['progress','转换进度'],['gpu','GPU 输出'],['human','人体原始']];
export function AtlasPage({tab}){
 if(tab==='progress')return <ProgressPage/>;
 if(tab==='gpu')return <GpuPage/>;
 if(tab==='human')return <HumanPage/>;
 return null;
}
