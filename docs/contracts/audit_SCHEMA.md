# 人工审计接口约定 v1（Claude 数据流水线 ⇄ 网页审计前端）

本目录是两边唯一的交接点。流水线（Claude）只写 `queue.jsonl`，前端（Codex 写的程序）只写 `decisions.jsonl` 和 `issues.jsonl`。两边都只追加、不改写、不删除。

## 1. queue.jsonl（流水线写：待审条目）
每行一个 JSON：
```json
{"item_id": "omnicontact/box/case1_carry/20260330000194_1_1775786896::unitree_g1_29dof",
 "clip_id": "omnicontact/box/case1_carry/20260330000194_1_1775786896",
 "robot": "unitree_g1_29dof",
 "source": "omnicontact",             // hiphi | omnicontact | grail | lafan_sample | ...
 "npz": "/home/arenalabs/umr_factory_data/out/<clip_dir>/<robot>.npz",
 "npz_sha256": "...",
 "format": "factory_npz_v1",          // 见 2；前端按此选择读取器，不认识的版本要明确报错
 "spec_version": "DATA_SPEC_v2",
 "frames": 5771, "fps": 30.0,
 "auto_qa": {"max_pen_ground_mm": 0.4, "max_pen_object_mm": 0.0, "max_pen_self_mm": 0.3,
             "solid_contact_rate": 0.97, "hover_rate": 0.01, "slip_p95_cm_s": 0.8,
             "flagged_frames": 12, "excluded_frames": 12},
 "priority": 0.83,                    // 0-1，越大越该先看（自动质检越差越靠前）
 "sample_kind": "risk",               // risk = 按风险挑的；random = 无偏随机抽样（用来估总体合格率）
 "enqueued_at": "2026-10-05T07:00:00Z"}
```

## 2. 数据格式（format 字段）
- `factory_npz_v1`：CPU 工厂输出，每个（片段, 机器人）一个 npz，字段见 `/home/arenalabs/umr_factory_code/README.md`，以及任意一个 `out/*/*.npz` 的 `files` 列表（qpos、root、各区域 ground/object 接触/间隙/打滑、`ground_point_contact`/`object_point_contact` 为 4096 点按位打包成 512 字节、穿透、flags、人体标签、物体位姿、provenance）。
- `gpu_npz_v2`：GPU 引擎输出（开发中），出来后在本文件追加字段说明。
- `packed_v1`：`/home/arenalabs/umr_dataset` 打包后的训练分片（catalog + shards）。
- 机器人静态资产：`/home/arenalabs/umr_factory_data/robots/<robot>/`（`<robot>.xml` + `mesh/`，`surface.npz`：4096 个表面点 points_local/normals_local/geom/body/region，`staging.json`）。
- 人体：`provenance.human_slots` 指向的 npz（`source_slots` = UMR 的 4096 个人体表面点，逐帧，世界系，要减去 `floor_offset_removed`）；SMPL-X 参数在 `provenance.source_data`。

## 3. decisions.jsonl（前端写：人工判决）
```json
{"item_id": "...", "npz_sha256": "...",       // 必须带审计时看到的文件哈希，文件变了判决作废
 "verdict": "accept" | "reject" | "needs_fix",
 "frame_ranges": [[120, 340]],                  // 可选：只针对这些帧段（闭开区间）；空 = 整条
 "reasons": ["penetration", "hover"],          // 取自下面的原因表，可多选
 "note": "左脚在 200 帧穿地",
 "reviewer": "lloyd", "ts": "2026-10-05T07:05:00Z", "ui_version": "x.y.z"}
```
原因表：`penetration_ground` `penetration_object` `self_collision` `hover` `slip` `contact_mismatch_human` `object_misaligned` `jitter` `unnatural_pose` `retarget_failure` `source_bad`（人体源本身有问题）`floor_wrong` `label_wrong` `other`。
同一 item 可以有多条判决，以最后一条为准。

## 4. issues.jsonl（前端写：系统性问题，比如"这台机器人所有片段左手都穿模"）
```json
{"scope": "robot" | "source" | "engine", "target": "fourier_gr3", "reasons": ["self_collision"],
 "note": "...", "example_items": ["..."], "reviewer": "lloyd", "ts": "..."}
```

## 5. 流水线怎么用这些判决（Claude 侧负责）
- `reject`：对应条目（或帧段）不进训练包；`needs_fix`：回到修复队列，带上原因重跑；`accept`：允许打包。
- 只有 `accept` 且 `npz_sha256` 与当前文件一致的条目才进训练包。
- 随机抽样（sample_kind=random）的合格率用来估计整批数据质量，决定是否放行整批。
