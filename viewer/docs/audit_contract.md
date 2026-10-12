# 人工判决存储与统计契约

本模块按远端 `audit/SCHEMA.md` v1 实现。`DATA_SPEC_v2.md` 和工厂 README 已完整阅读。它不修复数据、不重算标签、不生成自动通过或拒绝，不会在初始化或读取时创建判决文件。真实队列只有人在浏览器手动提交后才写入。自动化测试全部使用本项目 `tests/.audit-fixture-*` 临时目录，结束清理；不触及工厂正式审计队列。

## 服务端 API

```python
from backend.audit import AuditStore, sha256_file
store = AuditStore(audit_root, allowed_data_root=data_root)
record = store.append_decision(
    payload,
    {"item_id": item_id, "npz_sha256": viewed_snapshot_sha256,
     "frames": frame_count, "npz": absolute_npz_path},
    lambda: sha256_file(absolute_npz_path, allowed_root=data_root),
)
issue = store.append_issue(issue_payload)
records = store.read_records()  # {decisions: [...], issues: [...]}
stats = store.analytics(current_server_catalog_items)
```

`item_metadata` 必须是服务端已注册的审阅快照元数据，不能用表单提交字段覆盖。新打开或刷新页面可以显式建立新快照；提交时不能悄悄把旧快照重新绑定到新哈希。`current_hash_callback` 为无参函数，必须重新读取实际源文件，不能返回缓存中的旧 SHA。

判决必填 `item_id`、`npz_sha256`、`verdict`、`reasons`、`note`、`reviewer`、`ts`、`ui_version`；可选 `frame_ranges`，省略时规范化为空数组，表示整条。帧段是闭开区间 `[start, end)`，提交时检查 `0 <= start < end <= frames`。UTC 时间只接受有效日历时间，后缀 `Z` 或 `+00:00`。路由可在用户提交时补服务端 UTC 时间；核心存储不会静默补字段。

`verdict` 仅 `accept`、`reject`、`needs_fix`。原因严格取自明确原因表：

`penetration_ground`、`penetration_object`、`self_collision`、`hover`、`slip`、`contact_mismatch_human`、`object_misaligned`、`jitter`、`unnatural_pose`、`retarget_failure`、`source_bad`、`floor_wrong`、`label_wrong`、`other`。

SCHEMA 示例中的 `penetration` 不在明确原因表中，判为无效。系统性问题必填 `scope`、`target`、`reasons`、`note`、`example_items`、`reviewer`、`ts`；scope 仅 `robot`、`source`、`engine`。它没有 `ui_version` 或 `npz_sha256` 字段。两个记录类型都拒绝额外字段，保留唯一版本接口。

## 追加、并发与过期

目录各层与日志使用 `O_NOFOLLOW`，拒绝符号链接；相对受控目录的文件描述符防路径穿越。日志打开为 `O_APPEND`，取得 `flock(LOCK_EX)` 后验证所有历史行，再一次 `os.write` 写入 UTF-8 JSON 和换行，并对文件、目录 `fsync`。已有字节保持原样。坏 JSON、重复 JSON key、非有限数、空行、超过 1 MiB 的行、无结尾换行的截断行会报准确行号，阻止追加，绝不悄悄跳过或修复。

提交哈希必须同时等于请求中的哈希、正在审阅的快照哈希与实际源文件的新 SHA-256。持有追加锁后进行两次提交前核对，`fsync` 后再核对一次。源文件读取使用 1 MiB 缓冲，检查前后 inode、大小、mtime、ctime 和路径绑定，读取期间改写直接报过期。

外部数据生产者不共享网页审计锁，所以无法把源文件修改与两个独立文件的追加做成跨进程事务：若源文件恰在追加后改变，会抛出 `StaleSnapshotError(committed=True)`。此时旧哈希行已永久追加，必须保留；界面提示“判决已过期，需要重新审阅”，不能自动重试，也不能撤回或重写日志。提交前发现变化则 `committed=False`，不会追加判决。数据在成功返回后任何时刻重新生成，也会在下次刷新时被识别为过期。这是明示的并发边界，不代表通过了新数据。

异常为 `AuditValidationError`、`AuditLogError`、`StaleSnapshotError`，均有 `as_dict()`；最后一个还包含 `committed` 和 `reload_required`。操作系统的权限或符号链接错误也应由路由明确呈现。

## 统计含义

按同一 item 的物理追加顺序取最后一条，用户时钟不影响优先级。只有该条哈希与当前服务端目录的源哈希一致才为已审。未知当前哈希同样不能被算作已通过；过期判决计入待重审。

顶层与 `by_robot`、`by_source` 包括 `total`、`reviewed`、`pending`（含 stale）、`stale`、`partial_decisions`、`whole_item_reviewed`、`accepted`、`accept_rate`、`verdict_counts`。通过率的分母仅为有效的整条判决；局部帧段判决单列，不冒充整条数据通过。`by_item` 提供 `status`、`needs_review`、`stale`、`partial`、`verdict`、`decision`。拒绝原因排行只计最新有效的 `reject` / `needs_fix`，不重复累计被替代的历史判决。

`random_sample` 同时要求服务端来源元数据 `queued=true` 且原始队列 `sample_kind=random`。前端洗牌、随机浏览风险池、未入队扫描条目均不能改变这个资格。返回明确样本量 `n`、成功数 `successes`、95% Wilson 区间 `wilson_95`，n=0 时区间与比率为 null，`available=false`。只有最新有效整条判决进入 n；未审、过期和帧段判决数仍可见。

队列的 random 标签是生产者声明，网页未独立证明抽样总体、抽样设计、完成审阅过程无选择偏差。因此页面必须显示统计依据和限制，不能把已审风险条目的通过率或未完成随机样本宣称为全量总体质量。

## 独立只读校验

```sh
cd /home/arenalabs/motion_audit_web
/var/tmp/alphamotion/umr_factory/umr/env/bin/python check_decisions.py
# 可选按当前目录逐项重新计算哈希，仅核对每个 item 最新判决；旧历史允许哈希不同。
/var/tmp/alphamotion/umr_factory/umr/env/bin/python check_decisions.py \
  --catalog /home/arenalabs/umr_factory_data/audit/queue.jsonl --check-current
# issues 同样只读校验。
/var/tmp/alphamotion/umr_factory/umr/env/bin/python check_decisions.py \
  /home/arenalabs/umr_factory_data/audit/issues.jsonl --kind issue
```

脚本默认逐行 schema 校验；`--catalog` 额外提供帧数上界；`--check-current` 重新计算目录中源 NPZ 的哈希。历史坏行、最新哈希过期、缺失文件均非零退出。输出精确 `文件:行号`；`--json` 返回机器可读结果。不会写审计文件。

## 数据定义差异

DATA_SPEC_v2 的接触进入/离开阈值为 1 mm / 2 mm，打滑阈值 1 cm/s；工厂 README 的 Labels 段记载了 1 cm / 2 cm 与 3 cm/s。此模块不推测源标签实际版本、不改源数据。预览核对以 DATA_SPEC_v2 明确规则为尺子并标示界面现算量，保留生产者原始标签供人工比较。

## 验证记录

本地隔离测试：`python3 -m unittest discover -s tests -p test_audit.py -v`，20 项通过。覆盖 schema 枚举/必填字段/UTC、SHA-256、闭开区间、历史字节保持与单次写入、损坏日志拒写、精确行号、并发 4 进程 × 7 条追加、提交前/中/后的源文件变化、读取时变更、符号链接逃逸、issues、最后一条优先、过期/帧段/随机样本统计、Wilson n=0 与独立 CLI 当前哈希检查。

远端 `/var/tmp/alphamotion/umr_factory/umr/env/bin/python` 同样执行这 20 项测试，全部通过（0.311 秒，退出码 0）。依赖仅标准库，未修改已有环境。并发测试包含 4 个新建短寿命子进程，均已退出；测试仅创建项目内临时文件；本模块没有启动任何服务。

测试完成后只读检查正式 `audit/decisions.jsonl` 与 `audit/issues.jsonl`：两者均不存在，记录数均为 0，没有因自动化测试而生成正式人工判决。工厂 `out/*/*.npz` 现有 7 份；抽读踢球 GR3/G1 两份真实字段和 qpos 成功（5771 帧，实际 fps 为 90.00090026855469），与 DATA_SPEC_v2 的固定 30 fps 描述不同。播放应显示真实来源 fps，这一差异不通过写源文件“修复”。
