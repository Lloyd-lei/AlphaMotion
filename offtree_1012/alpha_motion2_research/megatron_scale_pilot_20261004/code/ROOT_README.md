# 修订交付摘要

适用的旧单帧配方已恢复到候选实现，101 项离线测试通过。真实训练尚未开始，真实数据补齐与 CUDA 验收仍待服务器恢复。

- 实际旧 run 配置已恢复，原字节 hash 与独立审计匹配。逐项差异见 recipe_diff.md，源码证据见 recipe_review.json。
- XWBC 恢复 InfoNCE、无条件重编码、16 类辅助 judge、真实足底穿透/接触几何、原六族 mask、八方向均值和优化器。相同 source row 在 InfoNCE 内去重，其他分支保留原样本权重。
- 主模型 25,743,428 参数；辅助 judge 266,512；合计 26,009,940。共享旧键加载375个，旧语义投影4键与旧judge4键弃用。新judge按当前已验证16体映射初始化，不猜旧类别。
- UniMate 仅同体 kinematic 任务；不造跨体配对/身体类别/metric floor。时间副批 world-slip/hpair 缺所需观测，明确关闭。新增 pure-SE3 是额外任务，不替换旧八方向主分支。
- 真实训练入口需要 clean-exit 的 full-loss CUDA validate receipt；代码、配置、数据、旧权重或运行环境变化即拒绝旧记录。配置 ready 开关不能绕过。

远端仍需：核对进程身份/退出与剩余容量；验收全部16体；对照原sole/contact来源补齐派生数据；审计UniMate全部真实帧FK；实际旧权重/Qwen与完整损失batch128 CUDA前后向；随后唯一有界pilot及配对评估。旧ArenaLabs连接未恢复，新服务器仍缺地址/登录用户。

本包不是下载全部完成、真实训练完成、仿真完成或真机验证完成的证据。监控断联记录仅本地保存，远端归档待恢复。
