# 可视化、测量、在线 Demo 与 surface 数据接入审计

本报告仅核查源码与已知数据格式，没有运行模型、重定向、动态仿真或线上上传。模块可复用性已落到文件和调用路径；工程代价是代码审阅后的估计，未当作实施工时。AlphaMotion-2 的联合 skeleton+surface 路线及局部区域文本锚定由主任务负责，本报告只提供可直接接入的工具和数据缺口。

**可立即利用的三个基础：** UniMate 的纯 NumPy LBS 可以把已有 rigged asset 和运动转成连续表面轨迹；UMR 的 surface slot 绑定和 overlay 能提供对应关系调试；AlphaMotion-1 已有基于每个机器人 MJCF 的离线物理一致性 QC。当前训练用骨架缓存本身仍缺 mesh 绑定和局部语义，不能把“可渲染”写成“已完成 surface 预训练”。

## 1. 模块级复用清单

|项目/模块|源码机制与可复用内容|工程代价|证据边界|
|---|---|---|---|
|[UniMate · 离线生成结果检查](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/utils/visualization.py#L30-L176)|反归一化后分别从RIC位置通道和旋转FK恢复关节；保存NPY、MP4、T-pose与文本标题。 **接入：**保留两种重建并列，量化位置头与FK一致性，不只看一条渲染轨迹。|低：接入统一motion envelope、fps与单位元数据。|可视化不是物理追踪；inference默认可关闭RIC，需显式开启双视图。|
|[UniMate · surface派生与渲染](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/mesh_animation/animate_lbs.py#L80-L349)|Blender只抽取rig/mesh/skin；纯NumPy按bind inverse、运动骨架和weights做LBS，可输出逐帧顶点NPZ。 **接入：**抽成离线SurfaceAdapter的参考实现；重用稳定face+barycentric采样形成跨帧material-point轨迹。|中：补三角化、骨骼裁剪/重排映射、canonical与world变换、分块缓存。|derived surface而非真实扫描或新dense对应标注；原素材必须确有rigged mesh。|
|[UniMate · 数据清洗](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/utils/skeleton.py#L468-L526)|根轨迹时间标准差、root-relative关节标准差、最活跃四分之一关节均值。 **接入：**作为低活动筛选并保留人工activity_keep例外。|低：批处理与阈值/版本写入manifest。|这是活动度，不是动作质量或稳定性。|
|[UniMate · 数据清洗](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/utils/motion_features.py#L506-L543)|逐帧平均关节位移除序列bbox最大extent；max/median比及绝对阈值共同识别断裂，拒绝非有限值。 **接入：**作为快速前置QC。|低：阈值可配置并存储原始统计。|单位为body/frame；不能直接报告厘米或每秒速度；extent还受root行程影响。|
|[UniMate website · 公开在线展示](https://linzhanmou.com/unimate/viewer.js?v=207)|interactive.js第64行导入viewer.js；GLTF/FBX/封装rig加载animation，AnimationMixer播放第一clip。 **接入：**借鉴同步多本体、mesh/skeleton切换、共享时间轴、显示文本。|中：自主结果manifest与model API、负载隔离需另接；网站素材协议单列。|已核查这条客户端链未见实时模型推理调用；画面groundAndNormalize会改变展示缩放和地面位置，不可拿来做物理QC。|
|[UMR · 对应与contact可视化](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/visualize_robot_retarget_result.py#L1742-L2043)|源/机器人surface slot颜色和重叠显示，地面/物体/自接触overlay，同步qpos回放；GLFW或Viser。 **接入：**最适合joint+surface一体viewer的调试层。|中：统一slot ids、人体/机器人/物体时间轴和有效性mask。|回放用mj_forward；不是控制闭环或真实机器人实验。|
|[UMR · 远端无桌面可视化](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/viser_mujoco_viewer.py#L525-L1184)|主机做MuJoCo运动学与Viser服务；浏览器渲染、截图和录屏。 **接入：**复用用于远端ArenaLabs数据审核。|低至中：依赖和内网接入、长序列流式装载。|无需把远端渲染当动态仿真。|
|[UMR · robot surface adapter](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L627-L712)|采集visual geoms成根坐标mesh；slot投影绑定face、geom id、局部位置与局部法向。 **接入：**复用robot rigid-link表面绑定，随后qpos经FK追踪material point。|中：visual/collision mesh区分、单位与根坐标保存。|slot跨本体对应由UMR训练得到，属于模型伪标签，不是人工GT。|
|[UMR website · 浏览器内真实计算](https://github.com/hanyang9/UMR/blob/b9d749900805fad4700df7ba9e229d57961be560/static/js/browser-umr-runtime-exact-v1.js#L138-L305)|4096 exterior surface samples→WASM correspondence训练→mesh binding→MuJoCo Jacobian+Clarabel约束优化→qpos。 **接入：**借鉴分阶段可观测进度、取消、几何/对应/动作overlay；可保留算法baseline。|中高：worker/WASM资产、cross-origin isolation、多线程、内存与Python一致性验收。|浏览器使用60epochs；Python有效默认500epochs；两者不可当同一耗时配置。|
|[UMR website · 可选服务器计算](https://github.com/hanyang9/UMR/blob/b9d749900805fad4700df7ba9e229d57961be560/static/js/mujoco-viewer.js#L2740-L2805)|默认startBrowserRetargeting；显式compute=server才打包robot文件/T-pose并POST /api/retarget/jobs。 **接入：**服务器异步任务协议可借鉴。|中高：当前website分支没有serve_umr.py；需找到真实后端实现或重建。|静态客户端存在分支不代表当前站点配置了在线server；本次未发起上传或执行。|
|[AlphaMotion-1 · 原研究交互viewer](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/viser_mesh_viewer.py:26)|由qpos FK得到真实G1 visual mesh逐帧pose；GT/model对照，human用capsule/line，slider/play。 **接入：**可重用robot mesh提取与轨迹对齐适配。|中：将G1 mapper、硬编码路径与人体capsule解耦到统一schema。|这是旧研究demo，不应直接标为当前Greenwich线上实现。|
|[AlphaMotion-1 · 拓扑通用预览](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/render_mesh.py:1)|每个bone独立free capsule，世界joint位置直接摆放；cm→m，Y-up→Z-up。 **接入：**适合无mesh样本快速fallback预览。|低。|capsule是显示代理，不是真实mesh监督，不能标记surface_available=true。|
|[AlphaMotion-1 · 接触伪标签](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/contact_label.py:252)|session floor校正、geometry gap和速度等规则；contact_low为免速度门槛训练分支，另存trust mask。 **接入：**可重用contact proxy和不可信mask；surface contact另派生时保持来源。|中：结合surface采样重新定义、保存阈值/坐标与置信度。|geometry heuristic不是触觉/接触力真值；unknown不是no-contact。|
|[AlphaMotion-1 · 本体条件离线物理一致性](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/physeval.py:376)|MJCF+世界qpos+fps→关节限制、接触滑移/穿透、自碰撞、jerk、CoM/ICP；逆动力学加摩擦锥可行接触力拟合、扭矩与floating-base residual。 **接入：**作为清洗基建最有价值的现成物理QC模块，保留每帧/汇总与vendor baseline。|中：schema与质量标志、robot-specific mass/friction/actuator版本、退化条件与性能批量化。|离线一致性分析，不是闭环sim tracking或sim2real成功率；未知执行器限值需mask。|
|[AlphaMotion-1 · human surface派生入口](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/export_smpl_params.py:27)|human_smpl本地rot6d→22body轴角、root_world厘米→米，写SMPL-X body params。 **接入：**接入既有SMPL-X模型，产出可追踪的body surface。|中：补原始betas/gender/hand参数及模板版本；缺失时明确default。|只导出body22，不恢复已丢失手指/形状；default shape是派生近似，不是原始人体mesh。|
|[AlphaMotion release · 现有产品服务复用](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/tool-audit/audit-source/AlphaMotion-release/src/alphamotion/service/app.py:111)|FastAPI warm pool + async job runner + SQLite结果记录；three LiveViewer负责源/目标/本体；play/timeline/jump异步任务和结果API，前端在assets/frontend。 **接入：**作为Alpha2服务壳和数据清洗结果审核入口，接入新的model adapter、surface overlay、eval结果。|中：新schema/模型装载/新增surface输出；保留现有接口且区分codebook版本。|源码核查，没有本次触发服务。重启会将unfinished jobs标failed，持久化的是记录和产物，不是任意任务续跑。|
|[AlphaMotion release · 现有live mesh展示](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/tool-audit/audit-source/AlphaMotion-release/src/alphamotion/viz/live.py:273)|set_trace载入robot qpos/mesh，set_smplx_skin可载入逐帧人体vertices/faces，源目标可链接相机与播放。 **接入：**在已有skeleton/robot/human mesh视图增加surface sample、region text、对应和contact overlay。|低至中：overlay缓存与frame/point有效mask；帧数据流式化。|mesh显示与ground-safe显示处理不是surface预训练或physics benchmark。|

## 2. 在线入口应如何准确描述

**UniMate：展示型客户端。** `interactive.js:64` 与 `index-viewer.js:11` 都导入 `viewer.js?v=207`。后者从模型文件取 animations，在 2075–2079 行选择第一条 clip 并由 `THREE.AnimationMixer` 回放；1196 行 fetch 的是 prompts.json。核查入口与 viewer 中未发现 text-to-motion/retarget 推理请求。这不否定项目的离线生成能力，只说明已公开部署的这条页面链展示已有结果。显示阶段的缩放、对齐地面和camera framing应与评测坐标独立保存，不能以展示后的视图直接算厘米误差。

**UMR：默认在浏览器内训练并重定向。** [入口导入](https://github.com/hanyang9/UMR/blob/b9d749900805fad4700df7ba9e229d57961be560/static/js/umr-studio-page.js#L182-L198) → [MuJoCo viewer](https://github.com/hanyang9/UMR/blob/b9d749900805fad4700df7ba9e229d57961be560/static/js/mujoco-viewer.js#L1-L7) → [exact runtime](https://github.com/hanyang9/UMR/blob/b9d749900805fad4700df7ba9e229d57961be560/static/js/browser-umr-runtime-exact-v1.js#L138-L305)。训练合约明确 [60 epochs、latent 1024、hidden 512、AdamW](https://github.com/hanyang9/UMR/blob/b9d749900805fad4700df7ba9e229d57961be560/static/js/umr-exact-training-kernels-v1.js#L119-L142)。它不是只播视频，亦不等于固定权重对新本体的直接推理。`compute=server` 是显式可选分支；当前 gh-pages 快照中没有 `serve_umr.py`，本次没有证明部署端 API 可用。

**AlphaMotion 发布工程：已补查现有服务与前端。** 远端 `/home/arenalabs/Desktop/AlphaMotion`，代码 HEAD `fd9a2badafa1a3c220667c0e861905354971e2f9`。其 [FastAPI service](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/tool-audit/audit-source/AlphaMotion-release/src/alphamotion/service/app.py:111) 启动 warm model pool 和三个 LiveViewer；play/timeline/jump 任务及结果API已存在，前端实际是 [assets/frontend/index.html](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/tool-audit/audit-source/AlphaMotion-release/src/alphamotion/assets/frontend/index.html:310)，仓库顶层 frontend 目录为空。UI调用任务API并轮询，与UniMate纯展示的页面不同。异步任务状态和结果持久化于SQLite/文件；未完成任务在服务重启后会标failed，不能描述为自动恢复执行。

[LiveViewer](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/tool-audit/audit-source/AlphaMotion-release/src/alphamotion/viz/live.py:273) 的 `set_trace` 支持robot mesh/qpos，`set_smplx_skin` 支持逐帧人体vertices/faces，已有源/目标同步与相机联动，适合直接加surface/region/contact overlay。保留原坐标评测和显示grounding之间的区别。

**当前codec与编辑器的版本边界：** [greenwich.py:15–22](/Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/tool-audit/audit-source/AlphaMotion-release/src/alphamotion/engine/greenwich.py:15) 明确新v2的megatron等使用132×13 FSQ代码，而现有timeline editor、Atlas、Equator仍用旧greenwich C1的256×20代码，等待重新索引。不能把整个在线编辑器一律标成megatron。源码可读证明实现存在，本次没有触发在线job或重新验证服务健康。

## 3. 骨架数据到 skeleton+surface 预训练的最小缺口

状态定义：**direct** 表示原始资产/记录直接提供；**derived** 表示依据绑定、FK或LBS确定性计算；**pseudo** 表示规则/模型推测；**missing** 表示没有可用监督。来源类型需逐字段保存，不能整条样本一概称为 ground truth。

|数据项|AlphaMotion-1 / XWBC 系列|UniML3D / UniMate|最小接入要求|
|---|---|---|---|
|骨架与运动|direct：已有 spec、pose/rot6d、root_world；robot qpos 的原始来源与映射需保留|direct：feature NPZ 保存 global_positions/local_rotations/root_facing_quat/fps；cond 保存 parents、offsets、T-pose、names等|统一单位、坐标、世界根轨迹、fps、joint映射与clip lineage；不要用归一化pos代替物理世界坐标|
|rest mesh / faces|机器人 MJCF visual geoms 可读取；human 可经 SMPL-X 生成；capsule只有 proxy|raw rigged FBX/GLB 可含mesh；Mixamo原animation-only FBX不含mesh，要关联人物rig；素材下载存在不等于全部clip已绑定|asset manifest → mesh版本/哈希/原始尺度；为每个clip建立到相应rig的可追溯引用|
|surface-to-body 绑定|robot：rigid link/geom id + local position/normal；human：SMPL-X拓扑与skin|raw export NPZ有skin_matrix；完整vertices/faces/bind matrices需从raw rig抽取；feature NPZ没有这些字段|两类绑定：rigid geom-local 或 LBS weights+inverse bind；绑定中保留原joint与清洗后joint映射|
|surface轨迹和法向|derived：robot FK；human SMPL-X body pose。缺少原shape/hand参数时只能标默认模板近似|derived：已有animate_lbs.py可输出逐帧vertices；rest面上固定face+barycentric采样得到连续material point|不要逐帧独立采点再当同一轨迹；保存rest采样种子、face id、barycentric、可见性/有效mask；法向由变形后mesh或正确变换导出|
|局部区域文字|现有joint-name文本embedding只覆盖关节标签，不等于掌面/指腹/脚底语义|clean_joint_names/caption不是dense region annotation|missing需补region→面/点membership、区域文本、左右/内外方向、来源和confidence；由skin权重向surface传播的joint标签只能先标pseudo|
|跨本体surface对应|现有paired motion不自动给vertex-to-vertex correspondence|同样不能从相似joint名称推定dense真值|UMR可产slot对应pseudo；另建少量人工region/landmark锚点与mask；几何重建和循环约束是训练约束，不伪装GT配对|
|接触/物体|existing contact_low/trust为geometry rule proxy；robot MJCF collision可计算proxy|feature格式不提供通用真实接触/物体SE(3)/触觉|保存接触双方region/point/object ids、距离/相对速度与时刻；无object运动不合成物体监督；触觉接口预留mask，不能填0当无触觉读数|
|动力学|每个robot自己的MJCF惯性、摩擦、actuator和控制器版本|大多艺术rig没有物理参数|保留embodiment条件与物理资产来源；艺术rig可以学运动学/surface语义，缺动力学时mask相关loss|
|dex hand|body22桥不产生指关节运动；有手mesh亦不等于有灵巧操作数据|逐asset审计finger DOF、skin和运动覆盖，不能因mesh有五指就声称有dex supervision|finger DOF、object pose、接触/抓取任务与末端SE(3)需独立manifest和有效mask|

原始 export 明确保存 `skin_matrix`：[blender_export.py](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/utils/blender_export.py#L1099-L1130)；训练用 feature NPZ 仅四类字段：[motion_features.py](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/utils/motion_features.py#L563-L590)。`cond.npy` 提供骨架/T-pose校准而非dense mesh：[数据格式](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/README.md#L545-L565)。LBS 支持多mesh、归一化weights、未驱动bones保持rest transform：[实现](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/data_process/mesh_animation/animate_lbs.py#L80-L299)。

## 4. 最小 adapter 合约（建议，尚未实现）

```text
MotionSample
  ids: corpus, source_sequence_id, motion_id, embodiment_id, asset_revision
  frames: timestamps, fps, world_from_canonical, length_unit, valid_frames
  skeleton: parents, names, rest_transforms, dof_spec, joint_map
  motion: qpos?, local_rotations, root_world, joint_positions
  surface_asset: vertices_rest, triangles, geom_or_link_ids?, inverse_bind?, skin_weights?
  surface_samples: face_id, barycentric, rest_point, local_normal, area_weight, seed
  regions: region_id, point_membership, text, source_type, confidence, valid_mask
  interactions: object_pose?, contact_pairs?, relative_velocity?, tactile?
  provenance: per_field {direct|derived|pseudo|missing, source, version, confidence}
  masks: skeleton, surface, region_text, contact, object, tactile, dynamics
```

geometry绑定独立缓存于每个asset，运动window只引用绑定并按需FK/LBS；不重复存储同一mesh在每个window中。初期需要原始/派生表面的少量golden clips核对骨架、bind pose和world transform；之后可按需计算表面window或离线分块缓存。模型看到的局部region语义允许先有粗区域，细表面归属逐渐补足；核心是把来源和未知mask保留下来。

## 5. 最短基础验收

1. **变换与绑定：** bind pose重建原mesh；rigid surface点与MuJoCo geom FK一致；LBS关键帧与Blender evaluated mesh数值对照。容差由单位/精度定，不写死未测值。
2. **跨阶段索引：** 骨骼裁剪、重排、parent合并后，skin权重和joint names指向同一个关节；原clip/fps/世界根轨迹保持可追踪。
3. **表面连续性：** 同一个face+barycentric在时间上是同一material point；变形后法向有限；缺mesh、缺region、缺contact不伪造零标签。
4. **评测坐标：** 先在原world单位计算QC，再做独立显示缩放；normalization/grounding变换必须可回溯。
5. **可重复和恢复：** 采样seed+资产hash+adapter版本决定缓存；中断不留下被误当完成的缓存；小样本可回放对比并存逐帧QC。

## 6. 复用授权的已查事实

- [UniMate仓库 LICENSE](https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/LICENSE) 是MIT；源码复用需保留copyright和permission notice。
- [UMR主分支 LICENSE](https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/LICENSE) 使用MIT式许可文本。
- UMR website是同一GitHub仓库的gh-pages快照；该快照根目录没有LICENSE，只找到coi-serviceworker及Three.js/MuJoCo/TFJS/WASM等vendor许可证。主分支许可与网站自有脚本的适用范围未单独查明，第三方文件沿各自license保留。
- UniMate官网client JS可下载审阅，但本次未证明这些文件属于其MIT主仓库。可参考交互设计；直接复制网站自有文件前补齐来源和许可记录。
- 代码许可与raw模型/动作/SMPL类数据许可分别记录，不能从源码MIT推定所有下载资产同许可。
