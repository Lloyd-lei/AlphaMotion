# 实验总索引（2026-08-07）

`/var/tmp/alphamotion/topo_exp/runs/` 下有 **130 个 run 目录**，其中真正承载结论的约 40 个，
其余是 smoke / sanity / 已被取代的早期版本。本文件按**问题**组织，不按编号——
编号是历史产物，问题才是我们真正在追的东西。

**命名规律**：`E*` = 空间层，`P1*` = 时序层第一代探索，`V5/V6/V7/V8` = 时序层架构世代，
`N3*` = lloyd 洞见快速迭代轮，`CTRL*` = 混淆对照，`RATE_*`/`S*` = 码本尺寸扫描。
后缀 `_0803`~`_0807` 是启动日期。

---

## 一、空间层（已冻结，不再迭代）

| 问题 | run | 结论 |
|---|---|---|
| FSQ 比特预算多少合适 | `E0a_b8s2/b10s2/b12s2/b16s1/b24s2` | 锁定 |
| 掩码预训练有没有用 | `E0b_mask25/mask50/sparse2/sparse30/final` | 有，锁定 |
| 单码本还是双码本 | `E0c_single` | 双 |
| 容量 scaling | `E19_K16/K96/K128`、`E20_K64/K96_L15` | **K=128 是拐点** |
| 数据量 scaling | `E21_data500/1000/2000/4000` | **4000 条无新增内在维度** |
| 参数量 scaling | `E24_small/mid/large` | 3.67M→18M 有效，18M→57.8M 无效 |
| 种子噪声带 | `E23_K32/K128_s1/s2` | ±0.9°，后修正为 2.2° |
| 自由度特征 | `E15_dof`、`E16_dof_k64`、`E25_dofhead/dofloss` | E25 失败（投影不可撤） |
| 合成具身增广 | `E22_synth`、`E26_fixed`、`E29_wristaug` | — |
| **语义编码器对决** | `E27_sig` vs **`E27_qwen`** | **Qwen 后投影质量赢 18/18** |
| **可见关节残差** | **`E28_resid`** | **被命令关节 6.99cm→0.00cm** |
| 判官头 | `E14_grad/obs` | 只学会人vs机器人，机器人间不分辨 |

**生产用**：`E19_K128_0805`（当前所有时序层的冻结底座，SigLIP 版）
**待迁移**：`E27_qwen_0806` + `E28_resid_0807` 的组合应进生产重训

## 二、生成层（早期探索，已搁置）

`E1A_arlm/v2`（离散 AR-LM）、`E1B_chunkdiff/vpred`（分块 latent diffusion）——
E1 对决未完成裁决，等时序层定案后重启。

## 三、时序层世代史（主线）

### P1：第一代，全局池化
`P1_oat_z`、`P1_unord_z`、`P1b_full/motion_vel/motion_w2/w5/n4`、
`P1c_C1_conv_*`、`P1c_C2_freq_*`
→ **全局池化只保住 22-73% 运动能量**；卷积臂和频率臂都输；确立 62.69 bit/s 预算

### V5：路径积分 + 时间码
`V5_pathint`、`V5_pathint_noaug`、`V5_baseline_frameCE`、
`V5b_full/Bonly/noC/cdrop`、**`V5c_0806`**（当时的最佳配方）
→ 引入路径积分损失、黎曼分段、条件 dropout、连续分数时间重采样

### N3：lloyd 三个洞见的快速迭代
`N3a_regAR`（寄存器级AR）、`N3b_xbody`（跨具身解码损失）、`N3c_root`（根速度）、
`N3f_varfloor`（输出方差下限）、N3e（再编码一致性，**训崩已弃**）
→ N3a 一度最佳；**content_share 指标被发现是抛硬币，多个判定作废**

### V6：把空间层的掩码搬到时间轴
`V6_mae`、`V6_mae_resid`、`V6_regar`
→ **证伪**：不治均值回归，两药合用最差；但**掩码确实剥离身体身份**（泄漏 0.119→0.067）

### V7：砍掉 OAT，两条臂决战
**`V7_jepa_0807`**（8内容+1速度，全局无序）
**`V7_maskfsq_0807`**（9内容，时间局部）
→ **maskfsq 3:2 胜**；**砍 OAT 是动作幅度修复的真因**

### V8：弧长坐标（**在跑**）
**`V8_arclen_0807`**（inv_mode=warp，对齐 jepa）
**`V8u_arclen_unif_0807`**（inv_mode=uniform，对齐 maskfsq）
→ maskfsq 的机器原样跑在**等弧长格子**上 + 强制速度 token。8+1，预算不变。

## 四、对照与扫描

| 组 | run | 目的 |
|---|---|---|
| **混淆裁决** | `CTRLa_maskfsq_4body`、`CTRLb_v5c_16body` | **已裁决：幅度=砍OAT，零样本=数据** |
| 词表率失真 | `RATE_jepa_l5`、`RATE_mf_l4/l3`（+ jepa l4/l3 排队中） | 词表是否偏大 |
| **码本尺寸搜索**（在跑） | `S0_arclen_4b`、`S1_speed37`、`S2_c85`、`S3_open88` | 速度码维度/内容码维度/开大让熵说话 |

## 五、结论文件（数字都在这，不用翻 run 目录）

`V7_verdict_0807.txt`、`CTRL_verdict_0807`、`V7_BRIDGE_verdict.txt`、
`EXAMA_verdict.json`（物理）、`EXAMB_verdict.txt`（码本统计）、
`EXAMC_verdict.txt`（六种输入范式）、`REMOTE_verdict.txt`（遥控器侦探）、
`V6_verdict_0807.txt`、`N3_*_verdict.txt`
**待生成**：`V8_verdict_0807`、`SIZE_verdict_0807`（训完自动出）

## 六、可以删的

约 30 个 `*_sanity`、`SMOKE_*`、`P1*_smoke_*`、`_smoke_*`，以及 08-02 那批
（`plain`/`v4_gradaccum`/`v5_global`/`v8_fsq`/`A_st_rfsq2`/`B_global_rfsq2`/
`RD_stage1/4`/`E2_perjoint`/`DS_*`/`SR_*`）——都是 TopoAE 时代的早期原型，
已被 E0/E19 系整体取代。**删前确认没有报告引用。**
