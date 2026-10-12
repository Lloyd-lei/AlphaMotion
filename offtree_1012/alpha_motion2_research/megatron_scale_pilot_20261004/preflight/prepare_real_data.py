import pathlib,sys,os,json,subprocess,datetime,fcntl,traceback,time
r=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'code'))
from remote_state import collect
from pilot_data import write_json
lock=(r/'data_preparation.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state={'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'start_ticks':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'boot_id':pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'cmdline':pathlib.Path('/proc/self/cmdline').read_bytes().replace(b'\0',b' ').decode(),'stages':[],'status':'running'}
write_json(r/'preflight/data_preparation_state.json',state)
def run(name,args):
 cap=collect();write_json(r/'preflight/current_remote_state.json',cap)
 need=cap['remaining_combined_bytes']+22*1024**3+100*1024**2
 if cap['free_home_bytes']<need:raise RuntimeError('combined capacity no longer fits all preparation budgets')
 state['active_stage']=name;write_json(r/'preflight/data_preparation_state.json',state);print('START',name,flush=True)
 result=subprocess.run([sys.executable,'-u',*map(str,args)],cwd=r)
 state['stages'].append({'stage':name,'returncode':result.returncode,'ended_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});write_json(r/'preflight/data_preparation_state.json',state)
 if result.returncode:raise RuntimeError(name+' exited '+str(result.returncode))
 print('DONE',name,flush=True)
try:
 m=json.loads((r/'data_xwbc/manifest.json').read_text())
 if not m.get('static_seal'):run('static_seal',[r/'code/seal_xwbc_static.py','--data-root',r/'data_xwbc','--source-cache','/var/tmp/alphamotion/topo_exp/cache_lafan_v3','--source-manifest-sha256','6eccf6a013d4e3d3e263468520d1b162889a282fd32af6115740d3d783583c29'])
 if not m.get('ground_recipe_seal'):run('ground_backfill',[r/'code/seal_xwbc_static.py','--data-root',r/'data_xwbc','--backfill-ground','--source-caches','/var/tmp/alphamotion/topo_exp/cache_lafan_v3','/var/tmp/alphamotion/topo_exp/cache_amass_lconv_tfix','--legacy-source-dir','/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo','--sole-json','/home/arenalabs/tmp/sole_offsets3.json'])
 run('unimate_fk_adapter',[r/'code/pilot_adapter.py','--root',r/'data_audit'])
 state['status']='completed';state['exit_code']=0
except Exception as e:
 state.update(status='failed',error=str(e),exit_code=1);traceback.print_exc()
finally:
 state['ended_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write_json(r/'preflight/data_preparation_state.json',state)
print(json.dumps(state),flush=True)
sys.exit(state['exit_code'])
