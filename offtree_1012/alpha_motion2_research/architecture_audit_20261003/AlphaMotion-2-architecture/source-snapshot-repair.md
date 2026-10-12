# 审计快照大小写冲突修复

发现 1 组大小写冲突，共 2 条记录：AlphaMotion1/v2_windows_GROUND.json 与 AlphaMotion1/v2_windows_ground.json。原 macOS 两个路径实际指向同一 inode，仅保留了大写文件内容，导致原检查中小写记录 SHA256 不匹配。

已从 ArenaLabs 逐一读取两个原文件，在本地独立编号目录保存，验证远端内容与原 manifest SHA256 一致后，为这两条记录增加 snapshot_rel。原 rel、remote、sha256 不变；其余 402 个快照文件与 manifest 记录未改动。旧快照路径保留但不再作为这两条记录的权威读取路径。

完整重算结果：404 / 404 条 verified；0 hash mismatch；0 missing；所有解析后的快照路径在 casefold 后唯一。已备份修复前 manifest。后续读取快照应优先使用 snapshot_rel，缺省才回退 source-snapshot/{tag}/{rel}。没有修改真实项目仓库或 Google Docs。
