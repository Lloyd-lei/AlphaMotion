# MCP 第一版验收 · 2026-10-08 UTC

15 个工具通过 SSH stdio 以标准 `initialize` / `tools/list` / `tools/call` 路径发现。项目 Codex 配置已被 `codex mcp get alphamotion` 读到，另提供通用 `.mcp.json`。本任务工具列表是否热加载取决于宿主重连；这里验证的是实际协议客户端，不冒称所有宿主 UI 都已验收。

- 本地 7 项测试通过：原生图像与指标、过期 hash、路径边界、持久任务与幂等、源代码变化拒绝、执行槽竞争、超时状态、视频实际解码、agent 观察与人工决定分离（部分断言在同一测试内）。回执 `mcp-unit-20261008.xml`。
- 实际远端目录返回 11 条动作；验收选择真实 GR3 踢球条目第 100 帧。`motion_inspect`、`motion_frame_metrics`、`motion_visualize` 均实际执行。
- 原始姿态采样点最低 z 为 −20.291 mm，修正后为 +0.554 mm；低于地面超过 0.5 mm 的采样点为 75 → 0。这只说明此帧此采样集，不能推出整条动作质量或连续网格无碰撞。
- MCP 返回原生 PNG 和逐帧来源报告。图像已打开检查，左右两栏为同一世界相机下的原始和修正姿态，帧号、时间、坐标和证据等级可见。
- 经 MCP 创建并提交 `foundation_smoke`，重复请求返回同一任务 ID。关闭 MCP 连接后另起连接，任务仍能查询，实际退出码为 0，报告 `passed=true`。没有启动训练。
- 完整协议与任务回执：`mcp-live-20261008/receipt.json`；图像：`mcp-live-20261008/real-motion-compare.png`。远端任务和视觉产物保留在 `agent_state/`。

仍未完成：将既有 UMR 重定向、下一轮训练/评估、动力学仿真变成输出隔离的 MCP 工作流。注册表将这些项标为 disabled 并给出原因。没有修改历史训练、人工审阅决定或原资源下载队列。

本次验证不宣称全部下载或内容校验已完成，不宣称训练效果、动力学仿真或真机效果。`foundation_tests` 已配置到真实 pytest 入口，但本次真实 MCP 作业验收执行的是 `foundation_smoke`。
