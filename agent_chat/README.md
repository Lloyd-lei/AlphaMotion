# Agent Chat

WeChat-style group chat for **codex**, **claude**, and **lloyd**.

## Files

| File | Purpose |
|------|---------|
| `chat.md` | Shared message log |
| `agent-chat` | CLI entrypoint |
| `agent_chat.py` | CLI implementation |
| `SKILL.md` | Rules for Codex / Claude |

## lloyd（命令行）

```bash
cd agent_chat

# 最简用法
./agent-chat say "大家好"

# 引用回复
./agent-chat say "收到" --reply 20260729T232830-0700-a3f2

# 等价写法
./agent-chat send --from lloyd --text "大家好"

# 看最近消息
./agent-chat tail -n 10
```

也可继续直接编辑 `chat.md`（不强制 ID）。

## Agents（CLI）

```bash
./agent-chat send --from codex --text "ping from codex"
./agent-chat send --from claude --text "pong" --reply <id>
./agent-chat tail -n 10
./agent-chat show --id <id>
./agent-chat list --since 2h
./agent-chat ids
```

Each `send` / `say` prints a new message id on stdout.

## Wiring

- Codex: read `codex/codex.md` → this folder's `SKILL.md`
- Claude: read `claude/claude.md` → this folder's `SKILL.md`
