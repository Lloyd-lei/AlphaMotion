export const meta = {
  name: 'surface-retargeting-plan',
  description: 'Read-only survey of all UMR code, derived-data regeneration cost and resource usage, then synthesize a consolidation + data + resource-yield plan',
  phases: [
    { title: 'Survey', detail: 'parallel read-only readers: 4 code groups, UMR-code sweep, 2 data-cost auditors, resource auditor' },
    { title: 'Synthesize', detail: 'one planner merges everything into a plan' },
    { title: 'Critique', detail: 'completeness critic checks the plan against the survey' },
  ],
}

const CTX = `
Context (machine of project owner "lloyd", Ubuntu 22.04, 32 threads, 91 GB RAM, one RTX PRO 6000 Blackwell 96 GB, one 1.8 TB NVMe: /home has only ~73 GB free, / has ~231 GB free; internet is a flaky USB WiFi ~3 MB/s).
Project: "UMR data factory" — retarget human motion (HiPHI, GRAIL, OmniContact, BONES-SEED, AMASS, LAFAN) to ~20 humanoid robots with strict per-point surface contact labels, built on the upstream UMR toolchain (hanyang9/UMR, local copy /var/tmp/alphamotion/umr_factory/umr/UMR at commit d1cdaed, env /var/tmp/alphamotion/umr_factory/umr/env).
lloyd now wants: (1) ALL UMR-related code merged into ONE niche repo named "surface-retargeting" (he typed "surfacere-retargeting"), organized like UMR's own toolchain; (2) to know which derived data cost us a lot of time to produce — data that cloud GPUs can regenerate quickly will NOT be preserved; (3) to free this machine's resources for a separate LLM project that has priority; the UMR project is not urgent and must yield.
Existing maps written 10-07 (may be stale, verify on disk): /home/arenalabs/umr_factory_data/platform_design/FOLDER_INDEX.md, map_factory.md, map_model_tools.md, map_research_repo.md, map_caches_artifacts.md, FINAL_DESIGN.md. Data spec: /home/arenalabs/umr_factory_data/DATA_SPEC_v2.md. Resource rules: /home/arenalabs/umr_factory_data/RESOURCE_LIMITS.md.
NOTE: on 10-11 /var/tmp/alphamotion/topo_exp and ~/tmp/cl_v3 were deleted (they were load-bearing training caches/contact labels). Treat them as gone.
UPDATE 10-12: the machine rebooted at 01:37-01:54 after the USB WiFi adapter reset; since then NO UMR downloaders, throttle_guard or factory jobs are running (they were not restarted on purpose, to yield to the LLM project). Partial BONES download (.incomplete) appears to be gone.
LOAD RULE (the WiFi dies under load): never run du/find over whole huge trees (miniconda3, ~/.cache, /var/tmp/alphamotion, raw/hiphi/extracted); use "du -sh --max-depth=1" with "timeout 120", reuse sizes from the 10-07 maps where possible, and keep each command short.
HARD RULES: strictly READ-ONLY. Do not modify, move, delete, chmod, git-init, or start/stop anything. No heavy CPU: prefix any du/find/python with "taskset -c 0-15 nice -n 10" and use timeouts; never run GPU work. Never use "pkill -f" or "pgrep -f". Never print secrets (HF token in ~/.cache/huggingface/token, ~/.git-credentials, ssh keys). Do not deserialize pickle files.
`

const COMPONENT = {
  type: 'object',
  properties: {
    path: { type: 'string' },
    what: { type: 'string' },
    status: { type: 'string', enum: ['production', 'active', 'experimental', 'dead', 'duplicate', 'third_party', 'data_not_code'] },
    size: { type: 'string' },
    approx_loc: { type: 'string' },
    entrypoints: { type: 'array', items: { type: 'string' } },
    depends_on: { type: 'array', items: { type: 'string' } },
    hardcoded_path_issues: { type: 'string' },
    exclude_from_repo: { type: 'string', description: 'large files, caches, venvs, outputs, secrets inside this path that must NOT go into git' },
    proposed_repo_location: { type: 'string', description: 'where it should live inside surface-retargeting/, or "leave out" with reason' },
  },
  required: ['path', 'what', 'status', 'proposed_repo_location'],
}
const CODE_SCHEMA = {
  type: 'object',
  properties: {
    components: { type: 'array', items: COMPONENT },
    cross_dependencies: { type: 'string', description: 'how these pieces import/call each other and other dirs' },
    risks: { type: 'array', items: { type: 'string' } },
  },
  required: ['components', 'cross_dependencies', 'risks'],
}
const DATA_SCHEMA = {
  type: 'object',
  properties: {
    products: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          path: { type: 'string' },
          what: { type: 'string' },
          size_gb: { type: 'number' },
          produced_by: { type: 'string', description: 'script/pipeline that made it' },
          local_cost: { type: 'string', description: 'measured from logs/timestamps where possible: wall hours, GPU hours, download hours, agent/human review effort' },
          irreproducible_inputs: { type: 'string', description: 'anything that cannot be re-fetched or re-run: manual QA decisions, hand-tuned configs, lost sources, gated downloads, deleted dependencies' },
          cloud_regen_estimate: { type: 'string', description: 'estimated time to regenerate on a cloud 8xH100-class node with fast network, with reasoning' },
          in_use_by: { type: 'string' },
          verdict: { type: 'string', enum: ['preserve', 'preserve_small_part', 'regenerate_on_cloud', 'drop'] },
          verdict_reason: { type: 'string' },
        },
        required: ['path', 'what', 'size_gb', 'local_cost', 'irreproducible_inputs', 'cloud_regen_estimate', 'verdict', 'verdict_reason'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['products', 'notes'],
}
const RES_SCHEMA = {
  type: 'object',
  properties: {
    gpu: { type: 'string' },
    processes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          pid_or_unit: { type: 'string' },
          what: { type: 'string' },
          project: { type: 'string', enum: ['umr_factory', 'alphamotion_other', 'lloyd_desktop_apps', 'other_session_do_not_touch', 'system', 'unknown'] },
          cpu_ram_gpu_net: { type: 'string' },
          recommendation: { type: 'string', enum: ['keep', 'pause', 'stop', 'stop_and_disable', 'ask_lloyd'] },
          how: { type: 'string', description: 'exact safe command (by PID file or systemctl unit), not pkill -f' },
          reason: { type: 'string' },
        },
        required: ['pid_or_unit', 'what', 'project', 'recommendation', 'reason'],
      },
    },
    disk: { type: 'string', description: 'free space per partition and what UMR-project items could be compressed/moved to give room, without deleting load-bearing data' },
    downloads: { type: 'string', description: 'state of HiPHI and BONES downloaders, remaining GB, and whether to pause them' },
  },
  required: ['gpu', 'processes', 'disk', 'downloads'],
}

const READERS = [
  { key: 'code-gpu-engine', schema: CODE_SCHEMA, prompt: `Survey the GPU UMR engine and the upstream UMR toolchain. Paths: /home/arenalabs/umr_gpu (umr_gpu/ package, eval/, production/, scripts/, README.md, _code_before snapshots under results dirs are NOT code), /var/tmp/alphamotion/umr_factory/umr/UMR (upstream, note what we changed vs upstream commit d1cdaed: run "git -C ... status --short" and "git diff --stat" read-only), /var/tmp/alphamotion/umr_factory/umr/tools, /var/tmp/alphamotion/umr_factory/umr/{configs,eval} , /var/tmp/alphamotion/umr_factory/lloyd_gpu_eval/*.py, /var/tmp/alphamotion/umr_factory/labels (pcl_contact proto). Describe UMR upstream's own directory layout and CLI/toolchain stages in detail so the new repo can mirror it. For each component give status, entry points, deps, hardcoded paths (count + examples), what must be excluded from git.` },
  { key: 'code-cpu-factory-dataset', schema: CODE_SCHEMA, prompt: `Survey the CPU reference factory and the dataset packer. Paths: /home/arenalabs/umr_factory_code (umrf package: staging robots, surface points, correspondences, run UMR, strict post-process, labels, QA; paths.py hardcodes paths; known bug objects.py treats box as sphere), /home/arenalabs/umr_dataset (UMRS v1 shard packer + loader). Also /home/arenalabs/umr_factory_data/robots/*/pipeline (per-robot QA pipelines, which scripts are code vs outputs) and /var/tmp/alphamotion/umr_factory/robots (registry robots.json, umr_configs, inventory scripts). For each component give status, entry points, deps, hardcoded paths, what to exclude from git, and where it belongs in a single repo that mirrors UMR's toolchain stages.` },
  { key: 'code-convert-ingest', schema: CODE_SCHEMA, prompt: `Survey the source ingestion and human-conversion code. Paths: /home/arenalabs/gpu_convert (GPU BVH->SMPL-X converter; runs/ and .inductor_cache are outputs/caches, not code), /var/tmp/alphamotion/umr_factory/hiphi2smplx, /var/tmp/alphamotion/umr_factory/h2s_shim, /var/tmp/alphamotion/umr_factory/beta_fit, /home/arenalabs/umr_factory_data/scripts, /home/arenalabs/umr_factory_data/scripts_grail (note .bak versions), /home/arenalabs/umr_factory_data/scripts_streamc, /home/arenalabs/umr_factory_data/workflows, /home/arenalabs/umr_factory_data/experiments/truncation/*.py, diversity census scripts under /home/arenalabs/umr_factory_data/diversity. Identify which are live vs superseded, licenses of third-party code (hiphi2smplx is MIT?), body-model files (SMPL-X .npz/.pkl model files must NOT go into git — find where they are and how code locates them).` },
  { key: 'code-audit-web', schema: CODE_SCHEMA, prompt: `Survey the audit web + MCP tooling and decide what part is UMR-specific. Path: /home/arenalabs/motion_audit_web (backend/: npzhead.py, progress.py, gpu_adapter.py, human_smplx.py, atlas_routes.py, adapters.py, app.py; frontend/src/atlas; agent_tools/ MCP; configs/mcp.json; scripts/service.py; tests/; .venv, .mcp-venv, cache/arrays, node_modules, dist are NOT code). It was originally a Codex-built generic motion audit app, extended for UMR outputs (gpu_npz_v2). Recommend whether it goes into surface-retargeting as a subpackage (e.g. viewer/ or audit/) or stays a separate repo, with reasons. Report frontend build requirements and size of source vs build outputs.` },
  { key: 'code-sweep', schema: CODE_SCHEMA, prompt: `Multi-angle SWEEP for UMR-related code that lives OUTSIDE these already-covered dirs: ~/umr_gpu, ~/umr_factory_code, ~/umr_dataset, ~/gpu_convert, ~/motion_audit_web, ~/umr_factory_data/{scripts,scripts_grail,scripts_streamc,workflows,experiments,diversity,robots}, /var/tmp/alphamotion/umr_factory/{umr,robots,hiphi2smplx,h2s_shim,beta_fit,labels,lloyd_gpu_eval}. Search (with nice + timeouts, excluding venvs/site-packages/node_modules/.git) in /home/arenalabs (incl. Desktop/"be water, robot" research repo which is a git repo — the real one has literal quote characters in the dir name, ~/tmp, ~/alphamotion_ctrl, ~/Desktop/arena-datagrid, ~/Desktop/AlphaMotion) and /var/tmp/alphamotion for: python files importing umrf / umr_gpu / UMR modules, files mentioning "umr_factory", "gpu_npz_v2", "surface.npz", "strict_seq", "UMR", scripts named *umr*, uniml3d_reload, sources/ inventory scripts, render scripts for UMR comparisons, /tmp/claude-*/ scratchpads holding UMR scripts that only live there. Only report things that belong to the surface-retargeting/UMR project (not Greenwich/AlphaMotion model code). For each, say whether it is a unique copy (would be lost) or a duplicate.` },
  { key: 'data-factory', schema: DATA_SCHEMA, prompt: `Audit regeneration cost of every DERIVED data product of the UMR factory (not the raw downloads, but do report raw sizes + how long downloads took and whether they are re-downloadable from public/HF sources with a token). Paths: /home/arenalabs/umr_factory_data/{raw,smplx,smplx_gpu,robots,corr,runs,out,qa,qa_renders,diversity,audit,production,experiments,pilot,hf_cache,logs}, /var/tmp/alphamotion/umr_factory/{umr_gpu/{problems,results,videos,robots,corr,runs,sources},robots,sources,labels,uniml3d_reload,lloyd_gpu_eval,beta_fit}, /home/arenalabs/umr_dataset/data, /home/arenalabs/gpu_convert/runs. Use logs (/home/arenalabs/umr_factory_data/logs, gpu_convert logs, grail logs) and file timestamps to MEASURE how long each product took (e.g. GPU converter throughput ~130 motion-hours per machine-hour; GRAIL conversion v1->v3 with ground calibration + collision caches; robot surface assets + correspondences + per-robot QA done by agents over days; truncation trial; diversity census with per-clip labels). Key question per product: is it cheap to regenerate on cloud GPUs, or does it encode slow/irreproducible work (manual QA decisions, tuned per-robot configs, license manifests, human review) that must be preserved? Small-but-precious items (robot assets, configs, manifests, label tables) should be preserve even if cheap in bytes.` },
  { key: 'data-legacy', schema: DATA_SCHEMA, prompt: `Audit the OTHER (non-UMR-factory) AlphaMotion data on this machine for regeneration cost, because lloyd wants to know what took a lot of time to produce and needs disk/resources for an LLM project. Paths: /var/tmp/alphamotion (all top-level dirs except umr_factory and the deleted topo_exp: xwbc_full 129 GB tarballs, xwbc_lafan, any_body, hf_cache, am2exp, ctrl_verify, config_harvest, robot_models, omniretarget, gmr_production*, gmr_spike, gmr_upstream, gem_*, genmo_*, studio, beyondmimic, boneseed_spatial_*, etc.), /home/arenalabs/tmp (largest subdirs), /home/arenalabs/alphamotion_artifacts, /home/arenalabs/alphamotion_ctrl (data/log dirs, and its 14 GB .git), /home/arenalabs/.cache/huggingface/hub (which models are used by what), /home/arenalabs/.cache/uv, /home/arenalabs/miniconda3/envs (which envs are used by running services or recent work). ALSO assess the damage of the 10-11 deletion of /var/tmp/alphamotion/topo_exp and ~/tmp/cl_v3: find what now has dangling symlinks or code path references (grep the research repo, which is the dir with literal quote chars: /home/arenalabs/Desktop/\\"be water, robot\\" ), and what it would take (inputs that still exist, scripts, compute) to rebuild the current training caches (amass_lconv_tfix, lafan_v3, C1 codes, text_align, contact labels v3) — on cloud vs locally. Give a verdict per item.` },
  { key: 'resources', schema: RES_SCHEMA, prompt: `Audit current resource usage on this machine so the UMR project can yield to a new LLM project. Use ps, nvidia-smi, free, df, ss -ltnp, systemctl --user list-units, docker ps (read-only). Known items: UMR downloaders (HiPHI: pid in /home/arenalabs/umr_factory_data/logs/hiphi_dl.pid; BONES: /home/arenalabs/umr_factory_data/logs/bones/dl.pid), throttle_guard (logs/throttle_guard.pid), motion_audit_web server on 8765 (2.2 GB RAM), user systemd units alphamotion-ardy-isaac (crash-loops ~600 restarts/h because a model xml is missing), alphamotion-motion-lab (port 2334), alphamotion-rollout-gallery (8891), greenwich-mobile / mobile_server on 8899 (DO NOT TOUCH), openclaw-gateway/node, docker container arena-datagrid-postgres, Cursor, Claude desktop, ChatGPT desktop, jupyter kernels, a Codex app-server (DO NOT TOUCH), other Claude sessions' processes (DO NOT TOUCH, e.g. pretraining_foundation acquire.py). For each notable process give the recommendation and exact safe command (PID file / systemctl --user unit / docker stop name). Report HiPHI/BONES download remaining GB (check raw/hiphi/data, logs/hiphi_streamc.log last events, raw/bones_seed/.cache incomplete size vs 45.47 GB) and whether to pause them to free bandwidth/disk for the LLM project (lloyd previously asked for both downloads; note the conflict). Report disk headroom an LLM project would likely need (model weights 15-150 GB) and where it fits.` },
]

phase('Survey')
const survey = []
for (let i = 0; i < READERS.length; i += 3) {
  const batch = await parallel(READERS.slice(i, i + 3).map(r => () =>
  agent(`${CTX}\n\nTASK (${r.key}):\n${r.prompt}\n\nReturn the structured result. Be concrete with paths, sizes and evidence; mark anything you could not verify as "unverified".`,
    { label: r.key, phase: 'Survey', schema: r.schema })
    .then(res => ({ key: r.key, res }))))
  survey.push(...batch)
  log(`survey batch ${i / 3 + 1} done`)
}
const ok = survey.filter(s => s && s.res)
const missing = READERS.map(r => r.key).filter(k => !ok.find(s => s.key === k))
if (missing.length) log(`survey readers that failed: ${missing.join(', ')}`)
const SURVEY_JSON = JSON.stringify(ok, null, 1)

phase('Synthesize')
const plan = await agent(`${CTX}

Below is the structured survey from ${ok.length} read-only readers${missing.length ? ` (FAILED readers, treat their areas as unverified: ${missing.join(', ')})` : ''}:
${SURVEY_JSON}

Write lloyd a PLAN (方案) in plain, short Chinese (no jargon walls, no English-Chinese mixing beyond unavoidable names/paths; he asked "说话简单精炼"). Nothing is executed yet; the plan is for his approval. Sections:
1. surface-retargeting 仓库: proposed directory tree mirroring UMR's own toolchain stages (ingest/convert human -> robot assets/surface points/correspondence -> retarget (GPU engine, CPU reference) -> strict post-process/contact labels -> QA/eval/render -> dataset packing -> audit viewer), mapping table old path -> new path, what stays OUT of git (venvs, caches, outputs, body-model files, third-party with incompatible license -> as submodule or fetch script), how the upstream UMR is included (submodule/vendored patch set — say which and why), how hardcoded paths get replaced (one config file / env var), how to keep currently running jobs working during migration (copy-then-switch, symlinks), git history choice, private GitHub repo under Lloyd-lei, and a migration order in steps with a rough time for each.
2. 数据: a table of derived data: size, how long it took us, what is irreproducible, cloud regen time, verdict (保留 / 只保留小部分 / 上云重算 / 丢弃). Make clear which small precious items must go into the repo or a private HF dataset (robot assets, configs, manifests, QA decisions), and which big items are simply regenerated on cloud. Include the honest status of what the 10-11 deletion broke and whether rebuilding is needed now or can wait for cloud.
3. 给 LLM 项目让路: concrete list of what to stop/pause now (with the exact safe command), what to keep, resulting free RAM/GPU/disk/bandwidth, and the conflict with the running downloads (recommend).
4. 需要 lloyd 拍板的事: numbered, each answerable in one line, with your recommendation.
Keep it tight: tables over prose. Do not invent numbers — use survey evidence or say "未核实".`,
  { label: 'planner', phase: 'Synthesize' })

phase('Critique')
const critique = await agent(`${CTX}

A planner wrote this plan from the survey below. Act as a completeness and correctness critic. Check: any UMR code path in the survey (or that you can find yourself on disk, read-only) missing from the repo mapping; any data verdict that would lose irreproducible work (e.g. dropping something that only exists here, or calling something "cheap to regenerate" when its inputs were deleted); any stop/pause command that would hit a DO-NOT-TOUCH process or uses pkill -f; numbers not backed by the survey; anything unclear for a non-native-jargon reader. Spot-check at least 5 claims on disk (read-only). Return a concise list of concrete corrections (each: what is wrong, evidence, fix), in Chinese, or "无问题" if none.

PLAN:
${plan}

SURVEY:
${SURVEY_JSON}`,
  { label: 'critic', phase: 'Critique' })

return { plan, critique, missing }
