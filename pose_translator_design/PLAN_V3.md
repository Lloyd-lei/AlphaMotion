# 训练计划 v3 — xwbc 数据底座版

日期:2026-08-04 · 触发:lloyd 发现 xwbc-retarget-data(XHugWBC 系,真机背书) · 生效条件:lloyd 对样片的质量终审通过
前置:PLAN(gate P3 已锁 tokenizer 配方)、R6_GATE_P3、XWBC_ASSESSMENT

---

## 一、数据底座(重定)

| 层 | 来源 | 内容 | 状态 |
|---|---|---|---|
| **金-A** | xwbc(138GB 下载中) | SMPL 人体 ↔ 11 台我们的机器人(+8 台赠品):~1.3万条 AMASS/台 + LAFAN 长连续段(至 4.4 分钟);qpos 与我们的 MuJoCo 模型 0.0mm 吻合;配对 100% | 待 lloyd 终审 |
| **金-B** | BONES-SEED | SOMA ↔ 厂商 G1,14.2万条 | 现役 |
| 辅 | OmniRetarget(G1 交互)、Adam-Pro(SOMA↔Adam,89GB 待验)、LAFAN 官方 | 单语语料,降权 |
| 退役 | 我们的 GMR 生产 | 质量被否;管线保留应急(toddy 等 xwbc 缺失机型) | 封存 |

关键性质:xwbc 自带 human_motion(22 关节 SMPL 世界运动学,同名配对)——人体侧升级为**双源**(SOMA + SMPL),对拓扑无关 tokenizer 是零成本第三骨架。LAFAN 长段正好满足时序需求(此前只有 ≤8 秒片段)。
注:本地 data/amass 不存在(旧记忆过期,已修正);xwbc 的 human_motion 使 raw AMASS 非必需。

## 二、阶段(并行两线)

**线 1:时序(不等新数据,已在跑)**
1. E1A-v2:10k 文件缓存(建设中)+ dropout + 最优点存档,重训 AR-LM
2. E1B:分块 latent 扩散(ARDY 组件)+ eval_rollout + guide_smoke(SE3 引导可行性)
3. E1 判决 → 锁生成层 → 接 ARDY 主干
4. 时序数据二期:xwbc LAFAN 长段入缓存(长视野训练)

**线 2:跨具身二世(xwbc 过审后启动)**
5. xwbc→缓存转换器(prep_gmr_cache 扩展:npz 关节名重映射 BFS→XML 序、human_motion SMPL 骨架 spec、Y-up 统一、直立性校验)
6. **tokenizer v2 训练/验证设计(lloyd 2026-08-04)——两阶段**:
   - **Phase A(科学)**:只用 xwbc 训练(SMPL+过审机器人),**BONES SOMA↔G1 作外部验证**——同时考动作分布迁移/人体骨架零样本(SOMA 78关节训练中全程缺席)/参数化迁移(g1_29dof→g1_30);双层验证:内层=xwbc留出片段(压缩质量),外层=BONES(跨骨架泛化),分开归因;SOMA手指单列记账(训练无手指拓扑,零样本大误差不冤)
   - **Phase B(产品)**:科学结论落袋后 BONES 并入训练,验证改两家各自留出片段
7. **E5/E9 正名战(Phase A 框架下空前干净)**:横轴=xwbc训练本体数(2→5→10→18),纵轴=固定外部考卷(BONES),考卷与训练零重叠;+ 比特×本体二维 + 对比权重缩放维度(修 E6 稀释混淆)+ 骨骼缩放增强(尺度外推)
8. 翻译器蒸馏重定向器:用 xwbc 配对训出的翻译器反哺 BONES-SEED→11 机器人(数据自增殖,GMR 正式退役)

**汇合点**:v2 tokenizer + E1 赢家主干 = 全量预训练(scale 阶段);之后约束阶梯(E3/E4+IK 投影)、WBC(G1,BeyondMimic 配方,可跟踪率终审)、E10-13 备用架按证据开闸。

## 三、立即执行清单

- [x] 138GB 全量下载(后台,~小时级)
- [ ] lloyd 样片终审(6 个 mp4 已推)
- [ ] E1A-v2(等 10k 缓存,今日)
- [ ] E1B 编写(今日)
- [ ] xwbc 转换器(过审后)
- 任务 #13 改写:SMPL 侦查目标已被 xwbc 满足,余项仅 Adam-Pro 验证

## 四、风险

- xwbc 质量终审可能局部否决(某些机器人/动作类)→ 按机器人子集采纳,配对性不受损
- XHugWBC 代码"coming soon"——管线参数未开源,数据固定;缓解 = §二.8 蒸馏路线(管线可从数据学出)
- LAFAN 许可(Ubisoft 非商用)——训练用途需在商业化前复核;AMASS 部分各子集许可同理
