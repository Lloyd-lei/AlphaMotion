# Greenwich-v2 P0/P1/P1.5 代码改动清单 (0830, 未提交)

全部改动默认关; 全关时训练与改前逐位一致 (同种子 40 步 tiny smoke,
改前/改后/再改后三次 eval_log 完全相同)。GPU 忙, 只做了 tiny smoke, 没跑 pilot。

## 新文件
- **exp_topo/v2_judge.py** — P0 判官。一张表四项预注册指标:
  幅度比(all/arm/leg, 复用 amplitude_report) / 上肢误差 mean+p95 cm (FK口径) /
  **腿角 髋+膝 geodesic mean+p95 deg (回归护栏)** / 方言逐维(FSQ 格码一致率 +
  量化前余弦, 编码套路照抄 zspace_battery)。
  固定卷子: `exp_topo/v2_windows.json` (LAFAN 片段级 val 惯例 rng(123) 取 7 片段,
  24 窗 x 90 帧, 种子固定落盘, 所有臂同卷)。reach 按 ckpt config 的 cache 列表
  算训练口径 (落盘 `v2_reach_cache.json`, 与训练日志逐位对上: 121.9/87.4/98.4)。
  方向: human→g1_29dof (全臂 holdout=真零样本) + human→gr3 (训练内对照)。
  已考: 生产 PT_c1dec_reenc_0812 (amp≈1.0, g1 腿角 15.0°/p95 28.8°) 和
  3.67M 参考 E24_small_0805 (腿角 17.8°/38.5°), 结果在各自 run 目录 v2_judge.json。
- **exp_topo/p0_pilots.sh** — 4 串行臂 (base / vf=var_floor 0.5 / fk=fk_loss / vffk),
  配方逐项照抄 E24_small_0805 (3.67M, 最便宜的 E 梯队配置); var_floor 0.5 =
  N3f_varfloor_0807 的既用值。每臂考判官, 汇总 runs/P0V2_results.json;
  末尾生产基线同卷补考一行。**没有执行** — GPU 空了再 `bash p0_pilots.sh`。

## 改动文件
- **exp_topo/train_dual.py** — 新开关 (全部默认关):
  - P1: `--var_norm` + `--var_norm_floor 1e-4` — 位置误差按关节数据位移 σ 归一。
  - P1.5: `--slip_pairs` + `--slip_w 0` — 1/4 batch 相邻行对副批, 两帧各解码,
    GT 脚踩地(水平脚速<0.10m/s)时罚解码脚水平速度平方 ((m/s)²), eval 行多打
    `slip_dec_ms`。相邻对查 sequences.json 片段表, 不跨片段、不碰 val 行;
    没有 sequences.json 的缓存 (如 cache_paired) 直接拒跑。
  两者都走自有 numpy RNG 流, 不碰 torch RNG。
- **exp_topo/dual_common.py** —
  - `run_dir(..., pos_w=None)`: var_norm 的逐关节乘法入口 (None=老路径)。
  - `pos_sigma_weights()`: σ 统计 + 地板 + 均值归一, σ 落盘缓存
    `<cache>/v2_sigma_<emb>_<N>.npz`。**实测发现的例外**: 根关节位置恒为原点
    (σ 精确=0), 打地板后权重 1e4 会吃光归一化质量 → 钉死关节 (σ<1e-6) 给中性
    权重 1、不参与归一。修后权重范围 0.12~12.7, 合理。
  - `slip_pair_starts()`: 片段内相邻训练行对表。
  - `amplitude_report(..., starts=None)`: 窗口可由调用方传死 (判官用);
    默认 None = 老抽样, amp_duel 不受影响。
- **exp_topo/semantics.py** — 修一个撞上的旧 bug: build_name_embeddings 未命中时
  会把整个 lut 缓存文件缩写成"这次要的名字"。改为合并写回、老值优先。

## 与计划的偏差 (如实记录)
1. **"held-out 窗口"只能做到片段级惯例**: train_dual 的 val split 是打散单行,
   拼不出连续窗口; 判官用 LAFAN 片段级 val (rng 123, 同 bridge_a.CodeData 惯例)。
   所有臂行级见过这些帧的程度相同; human→g1 对每条臂含生产都是留出具身。
2. **var_norm 是 pos_j × 归一化的 1/σ**, 不是裸除: 裸除会把位置项整体放大
   ~10 倍 = 混进学习率变量; 沿用 arm_boost 的 mean-1 纪律。外加上面说的
   钉死关节例外。
3. **slip 的踩地判据在根相对坐标系**: 缓存 pose 无世界根 (root_world 只有少数
   具身有), 用根相对 FK 脚速 <0.10 m/s 是能拿到的最忠实版本。
4. **判官上肢/幅度全走 FK-of-rotations**, 不读 hpos (渲染只看旋转头, 0812 审计)。

## ⚠ 需要 lloyd 决定的一件事 (本 session 的现场发现)
semantics 的 clobber bug 在今天判官首跑时把 `/var/tmp/alphamotion/topo_exp/`
`qwen3_names.pt` 冲剩 88/380 键 (siglip 也被今天的 smoke 重写过一遍, 值有 3e-4
级重编码漂移; lafan 覆盖没丢)。缓存不全 = qwen3 配方训练加载时消耗 RNG =
FIX split 复现约定被破坏。已生成合并恢复文件 (老值优先, **没动现有文件**):
`qwen3_names.merged.pt` (380键) / `siglip_names.merged.pt` (300键)。确认后:
`cd /var/tmp/alphamotion/topo_exp && cp qwen3_names.pt qwen3_names.pt.clobbered && mv qwen3_names.merged.pt qwen3_names.pt && mv siglip_names.merged.pt siglip_names.pt`
(另: cache_paired/soma77 的 106 个名字在改前的 live+bak 里就都不在 —— 旧账, 与本次无关。)
