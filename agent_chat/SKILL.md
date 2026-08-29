---
name: agent-chat
description: >-
  Post and read messages in the project group chat (agent_chat/chat.md).
  Use when communicating with other agents or lloyd via the shared chat file.
---

# Agent Chat Skill

## Paths

- Chat file: `agent_chat/chat.md`
- CLI: `agent_chat/agent-chat`

Run all commands from the project root, or use the full path to `agent-chat`.

## Mandatory rules

1. Before posting or replying, read recent context:
   ```bash
   ./agent_chat/agent-chat tail -n 20
   ```
2. **Never** use Write/Edit tools to append messages in `agent_chat/chat.md`. Always use `send`.
3. To reply with a traceable reference, use `--reply <id>` where `<id>` is a prior message id.
4. To inspect one message before replying:
   ```bash
   ./agent_chat/agent-chat show --id <id>
   ```
5. After `send`, stdout prints the new message id — keep it for later references.

## Identity

- Codex must use `--from codex`
- Claude must use `--from claude`
- **lloyd** uses `./agent_chat/agent-chat say "..."` or `send --from lloyd` (agents must not impersonate lloyd)

## Commands

```bash
# lloyd 快捷发消息
./agent_chat/agent-chat say "hello everyone"
./agent_chat/agent-chat say "收到" --reply 20260729T232830-0700-a3f2

# Post a message (any participant)
./agent_chat/agent-chat send --from codex --text "hello"
./agent_chat/agent-chat send --from lloyd --text "hello"

# Reply with reference
./agent_chat/agent-chat send --from claude --text "got it" --reply 20260729T232830-0700-a3f2

# Recent messages
./agent_chat/agent-chat tail -n 20

# One message by id
./agent_chat/agent-chat show --id 20260729T232830-0700-a3f2

# Filtered list
./agent_chat/agent-chat list --from codex --since 2h

# All ids
./agent_chat/agent-chat ids
```

## Message format (written by CLI)

```markdown
---

**codex** · 2026-07-29 23:28:30 (UTC-7)
`id:20260729T232830-0700-a3f2`

message body

---
```

Do not hand-write `id:` lines. The CLI generates timestamps and ids automatically.
