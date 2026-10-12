# AlphaMotion-2 训练采样策略：源码核查与建议

结论：建议以同步 skeleton/surface 连续短窗口作为二代主训练单位，保留单帧/静态对应辅助。这里的“往后采样”需要拆开理解：连续窗口取数据、窗口内双向建模、因果下一步预测不是同一件事。此结论是实验建议，并未锁定生成器、量化方式或训练配置。

## 已核实的源码事实

**AlphaMotion-1 主FSQ路径**：当前推荐megatron的tdec=0；主批从训练行随机抽独立帧。存在接触均衡采样分支，但仍抽行。

[verified-megatron-config.json:59-65](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/verified-megatron-config.json:59>)；[train_dual.py:358-371](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:358>)。

**AlphaMotion-1 数据可复用性**：sequences.json的start/len提供片段边界，代码据此找同片段相邻帧/窗口；root_world.pt在root_pos路径中加载。不能仅凭帧数或行号假定时间顺序、fps和root齐全。

[dual_common.py:610-638](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/dual_common.py:610>)；[tdec.py:25-48](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/tdec.py:25>)；[train_dual.py:276-299](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:276>)。

**AlphaMotion-1 独立时序实验**：tdec>1可把K帧窗口压平后在解码器融合相邻码；当前megatron未启用。train_latent_diffusion.py另取64帧默认窗口，其中末8帧是噪声目标chunk，前56帧为干净上下文；Transformer调用没有因果attention mask。

[train_dual.py:149-158](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:149>)；[train_latent_diffusion.py:32-53](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_latent_diffusion.py:32>)；[train_latent_diffusion.py:78-115](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_latent_diffusion.py:78>)。

**UniMate 窗口**：已审默认配置T-pose条件，max_motion_length=60；长片段均匀随机选择合法起点，截取连续60帧，短片段补零并使用有效长度mask。first_frame条件分支会多取1帧作为条件且默认从0开始。

[configs/uniml3d_60frames_graph_adaln.json:18-38](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/configs/uniml3d_60frames_graph_adaln.json#L18-L38>)；[unimate/dataset/transforms.py:17-62](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/dataset/transforms.py#L17-L62>)；[unimate/dataset/transforms.py:114-124](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/dataset/transforms.py#L114-L124>)；[unimate/dataset/mixture/dataset.py:856-906](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/dataset/mixture/dataset.py#L856-L906>)；[unimate/dataset/mixture/collate.py:168-182](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/dataset/mixture/collate.py#L168-L182>)。

**UniMate 时间尺度**：Dataset要求30fps，显式非30fps报错；旧feature缺fps字段会警告并按30fps假设。特征中的速度为每帧差分，不能直接混用不同fps。60帧约2秒。

[unimate/dataset/mixture/dataset.py:629-651](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/dataset/mixture/dataset.py#L629-L651>)；[unimate/dataset/mixture/dataset.py:71-79](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/dataset/mixture/dataset.py#L71-L79>)。

**UniMate 生成方式**：窗口内每关节的时间注意力只按有效长度mask，不屏蔽未来帧；从[B,J,D,T]整块噪声开始，ODE或DDPM迭代共同生成整个窗口，不是逐帧因果自回归。整段生成也不等于一次网络forward。

[unimate/models/denoiser/graph_adaln.py:72-104](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/graph_adaln.py#L72-L104>)；[unimate/models/denoiser/blocks/graph.py:222-241](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/models/denoiser/blocks/graph.py#L222-L241>)；[unimate/inference/generate.py:101-125](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/unimate/inference/generate.py#L101-L125>)。

**UMR**：训练样本是canonical静态点云，训练表面对应；运动阶段逐帧优化qpos，用前帧warm start，可加一阶差分与二阶差分正则以及后置LQR。它不是学习时序运动生成器。

[scripts/train_correspondence_template_residual_ae.py:217-241](<https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L217-L241>)；[scripts/train_correspondence_template_residual_ae.py:650-669](<https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/train_correspondence_template_residual_ae.py#L650-L669>)；[scripts/retarget_smpl_to_humanoid_surface_vector.py:2700-2751](<https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2700-L2751>)；[scripts/retarget_smpl_to_humanoid_surface_vector.py:1910-1918](<https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L1910-L1918>)；[scripts/retarget_smpl_to_humanoid_surface_vector.py:2826-2848](<https://github.com/hanyang9/UMR/blob/d1cdaedaf67a827198da71c5fa831f48b2e51f91/scripts/retarget_smpl_to_humanoid_surface_vector.py#L2826-L2848>)。

**评测划分**：Alpha1旧默认split_indices随机划分帧；tdec源码明确注明一些val窗口其余帧可能来自训练行。UniMate已审默认test_split_ratio=0。二代必须重新以完整source动作组及本体留出来评测，不照搬这些配置作为严谨泛化验证。

[dual_common.py:237-242](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/dual_common.py:237>)；[train_dual.py:328-335](</Users/yudonglei/.codex/visualizations/2026/10/04/01a10493-9b02-7710-8b73-b474e3192aeb/alphamotion-architecture/source-snapshot/AlphaMotion1/train_dual.py:328>)；[configs/uniml3d_60frames_graph_adaln.json:18-38](<https://github.com/Friedrich-M/UniMate/blob/2c5b384715aa63d8639b1ed7eb74bfe614570c7a/configs/uniml3d_60frames_graph_adaln.json#L18-L38>)。

## 最小实验建议（尚未执行）

1. 主训练单位建议为同步skeleton+surface的连续短窗口；继续使用Alpha1原数据，通过片段索引重新组织读取，而不是因为单帧训练过就丢弃数据。单帧重建/映射和静态表面对应保留作辅助任务。
2. 先固定一个候选表示与骨干，做T=1、16、60三个时间上下文对照（30fps时约0、0.5、2秒上下文），固定完整动作划分，按总有效帧数/计算预算报告可比结果；不同时更换FSQ和生成器，避免归因混乱。T=1表示单帧，不表示0秒采样间隔。
3. 窗口长度优选后，再比较局部surface区域文本：正确文本、无文本、打乱区域文本，并保留surface-only和skeleton-only消融；完整主模型仍联合skeleton+surface。用完整未见本体留出衡量核心thesis，不能把时间平滑提升当成zero-shot证明。
4. 单帧/静态辅助可从同一有效窗口抽帧，或使用只有静态mesh的资产；训练权重与混合比例先不锁定，必须逐任务报告有效样本数和损失归一化。
5. 先比较单帧精度/EE误差分位数、表面对应/穿透、接触保持和世界系滑移、速度/加速度/jerk及窗口边界抖动。离线动力学、闭环sim和真机证据分开。
6. 是否连续窗口、是否因果预测、是否FSQ量化、是否flow/diffusion是不同设计轴；保留FSQ时也可以加时序encoder/decoder或学习token/latent序列生成，未授权锁定最终架构。

若“往后采样”指随机起点后取一段连续帧，建议采用；若指每步只能读过去、逐帧预测未来，那是另一项因果部署选择。离线retarget可读源动作整段；在线低延迟控制应另测因果或有限look-ahead，不把训练时可见未来直接当在线可用。

## 复用现有数据的必要条件

- 时间：保留source sequence id、原始frame index或timestamp、fps/重采样映射；窗口不跨片段，skeleton/surface/contact/物体必须对齐同一时刻。缺字段时先核对原缓存，不能按行号猜。
- 世界运动：保留root position/orientation trajectory和world↔canonical变换、尺度与轴约定；用真实dt算导数。仅有root-relative姿态不足以判断世界系接触滑移。
- 表面：对同一窗口追踪相同mesh物质点；rigid body用固定link-local点，LBS用固定face+barycentric/bind/weights，避免每帧随机重新采点。区域文本需与该版本mesh的membership对应且带来源。
- 缺模态：frame/joint/point有效性与模态availability分开；缺mesh/contact/tactile/region不是零值真值。允许局部可用数据训练相应任务，不伪造密集监督。
- 划分：先按完整原始动作/主体等source group划train/eval，再抽窗口；同一动作映射到20个本体的派生样本共用source group。再单独设完整本体holdout，报告seen-body新动作与unseen-body结果。

本次仅审本地固定源码快照和已核实megatron配置；没有逐项打开远端训练缓存，未证明每个缓存都有完整timestamps、root轨迹和mesh绑定。未修改模型/配置/Docs、未训练、未SSH修改数据。

Alpha1原实现根：/home/arenalabs/Desktop/be water, robot/pose_translator_design/exp_topo；文中本地链接指向本次审计快照。
