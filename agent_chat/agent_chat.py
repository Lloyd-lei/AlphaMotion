#!/usr/bin/env python3
"""Agent Chat CLI — append/read WeChat-style messages in chat.md."""

from __future__ import annotations

import argparse
import fcntl
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

AGENTS = frozenset({"codex", "claude", "lloyd"})
MESSAGES_MARKER = "<!-- agent-chat:messages -->"
ID_LINE_RE = re.compile(r"^`id:(?P<id>[^`]+)`\s*$")
HEADER_RE = re.compile(
    r"^\*\*(?P<sender>[^*]+)\*\* · (?P<timestamp>[^\n]+?)(?: · ↩ `(?P<reply>[^`]+)`)?\s*$"
)
ID_ANYWHERE_RE = re.compile(r"`id:([^`]+)`")


@dataclass
class Message:
    msg_id: str
    sender: str
    timestamp: str
    reply_to: str | None
    body: str
    raw: str


def chat_path() -> Path:
    return Path(__file__).resolve().parent / "chat.md"


def local_tz() -> timezone:
    return datetime.now().astimezone().tzinfo or timezone.utc


def format_tz_offset(tz: timezone) -> str:
    now = datetime.now(tz)
    offset = now.utcoffset()
    if offset is None:
        return "+0000"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    hours, rem = divmod(total, 3600)
    minutes = rem // 60
    return f"{sign}{hours:02d}{minutes:02d}"


def format_tz_label(tz: timezone) -> str:
    offset = format_tz_offset(tz)
    if offset in {"+0000", "-0000"}:
        return "UTC"
    sign = offset[0]
    hours = int(offset[1:3])
    minutes = int(offset[3:5])
    if minutes:
        return f"UTC{sign}{hours}:{minutes:02d}"
    return f"UTC{sign}{hours}"


def make_msg_id(now: datetime | None = None) -> str:
    tz = local_tz()
    now = now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)
    offset = format_tz_offset(tz)
    suffix = f"{random.randint(0, 0xFFFF):04x}"
    return f"{now.strftime('%Y%m%dT%H%M%S')}{offset}-{suffix}"


def format_header_time(now: datetime) -> str:
    tz = local_tz()
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({format_tz_label(tz)})"


def ensure_chat_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        """# Agent Chat

参与者：**codex** · **claude** · **lloyd**

- 所有人通过 `./agent-chat` CLI 发消息（lloyd 可用 `./agent-chat say "..."`）
- 也可直接编辑本文件发言
- 消息 ID 格式：`id:YYYYMMDDTHHMMSS±HHMM-xxxx`

<!-- agent-chat:messages -->

""",
        encoding="utf-8",
    )


def read_chat_text(path: Path) -> str:
    ensure_chat_file(path)
    return path.read_text(encoding="utf-8")


def split_messages(text: str) -> tuple[str, list[str]]:
    if MESSAGES_MARKER in text:
        prefix, rest = text.split(MESSAGES_MARKER, 1)
        prefix = prefix + MESSAGES_MARKER + "\n"
    else:
        prefix = text
        if not prefix.endswith("\n"):
            prefix += "\n"
        rest = ""
    chunks = [c for c in re.split(r"\n---\n", rest) if c.strip()]
    return prefix, chunks


def parse_message_chunk(chunk: str) -> Message | None:
    lines = chunk.strip().splitlines()
    if len(lines) < 2:
        return None
    header = lines[0].strip()
    match = HEADER_RE.match(header)
    if not match:
        return None
    msg_id = None
    body_start = 1
    if body_start < len(lines) and (id_match := ID_LINE_RE.match(lines[body_start].strip())):
        msg_id = id_match.group("id")
        body_start += 1
    while body_start < len(lines) and not lines[body_start].strip():
        body_start += 1
    body = "\n".join(lines[body_start:]).strip()
    if not msg_id:
        return None
    return Message(
        msg_id=msg_id,
        sender=match.group("sender").strip(),
        timestamp=match.group("timestamp").strip(),
        reply_to=match.group("reply"),
        body=body,
        raw=chunk.strip(),
    )


def parse_messages(text: str) -> list[Message]:
    _, chunks = split_messages(text)
    messages: list[Message] = []
    for chunk in chunks:
        msg = parse_message_chunk(chunk)
        if msg:
            messages.append(msg)
    return messages


def parse_since(value: str) -> datetime:
    value = value.strip().lower()
    now = datetime.now(local_tz())
    if value.endswith("h"):
        hours = float(value[:-1])
        return now - timedelta(hours=hours)
    if value.endswith("m"):
        minutes = float(value[:-1])
        return now - timedelta(minutes=minutes)
    if value.endswith("d"):
        days = float(value[:-1])
        return now - timedelta(days=days)
    raise argparse.ArgumentTypeError(f"invalid duration: {value!r} (use e.g. 2h, 30m, 1d)")


def parse_msg_datetime(timestamp: str) -> datetime | None:
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", timestamp)
    if not match:
        return None
    dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=local_tz())


def render_message(
    sender: str,
    msg_id: str,
    body: str,
    now: datetime | None = None,
    reply_to: str | None = None,
    ref_preview: str | None = None,
) -> str:
    now = now or datetime.now(local_tz())
    header = f"**{sender}** · {format_header_time(now)}"
    if reply_to:
        header += f" · ↩ `{reply_to}`"
    lines = [header, f"`id:{msg_id}`", ""]
    if reply_to:
        preview = (ref_preview or "").strip()
        if preview:
            preview = preview.replace("\n", " ")
            if len(preview) > 120:
                preview = preview[:117] + "..."
            lines.append(f"> ref `{reply_to}`: {preview}")
        else:
            lines.append(f"> ref `{reply_to}`")
        lines.append("")
    lines.append(body.rstrip())
    return "\n".join(lines)


def append_message(path: Path, block: str) -> None:
    ensure_chat_file(path)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.seek(0)
            content = fh.read()
            if MESSAGES_MARKER not in content:
                if not content.endswith("\n"):
                    content += "\n"
                content += MESSAGES_MARKER + "\n"
            trimmed = content.rstrip("\n")
            if trimmed.endswith(MESSAGES_MARKER):
                new_content = trimmed + "\n\n---\n\n" + block + "\n"
            else:
                new_content = trimmed + "\n\n---\n\n" + block + "\n"
            fh.seek(0)
            fh.truncate(0)
            fh.write(new_content)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def cmd_send(args: argparse.Namespace) -> int:
    sender = args.from_agent
    if sender not in AGENTS:
        print(f"error: --from must be one of {sorted(AGENTS)}", file=sys.stderr)
        return 2

    path = chat_path()
    text = read_chat_text(path)
    messages = parse_messages(text)

    reply_to = args.reply
    ref_preview = None
    if reply_to:
        target = next((m for m in messages if m.msg_id == reply_to), None)
        if target is None:
            print(f"warning: reply target not found: {reply_to}", file=sys.stderr)
        else:
            ref_preview = target.body

    msg_id = make_msg_id()
    block = render_message(
        sender=sender,
        msg_id=msg_id,
        body=args.text,
        reply_to=reply_to,
        ref_preview=ref_preview,
    )
    append_message(path, block)
    print(msg_id)
    return 0


def print_message(msg: Message) -> None:
    print(msg.raw)
    print("---")


def cmd_tail(args: argparse.Namespace) -> int:
    messages = parse_messages(read_chat_text(chat_path()))
    for msg in messages[-args.n :]:
        print_message(msg)
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    messages = parse_messages(read_chat_text(chat_path()))
    target = next((m for m in messages if m.msg_id == args.id), None)
    if target is None:
        print(f"error: message not found: {args.id}", file=sys.stderr)
        return 1
    print_message(target)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    messages = parse_messages(read_chat_text(chat_path()))
    since_dt = parse_since(args.since) if args.since else None
    filtered: list[Message] = []
    for msg in messages:
        if args.from_agent and msg.sender != args.from_agent:
            continue
        if since_dt is not None:
            msg_dt = parse_msg_datetime(msg.timestamp)
            if msg_dt is None or msg_dt < since_dt:
                continue
        filtered.append(msg)
    for msg in filtered:
        preview = msg.body.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        reply = f" ↩ {msg.reply_to}" if msg.reply_to else ""
        print(f"{msg.msg_id}  {msg.sender}{reply}  {preview}")
    return 0


def cmd_ids(_: argparse.Namespace) -> int:
    messages = parse_messages(read_chat_text(chat_path()))
    for msg in messages:
        print(msg.msg_id)
    return 0


def cmd_say(args: argparse.Namespace) -> int:
    args.from_agent = "lloyd"
    return cmd_send(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-chat", description="Agent Chat CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    send = sub.add_parser("send", help="append a message")
    send.add_argument("--from", dest="from_agent", required=True, choices=sorted(AGENTS))
    send.add_argument("--text", required=True, help="message body")
    send.add_argument("--reply", help="message id to reply to")
    send.set_defaults(func=cmd_send)

    say = sub.add_parser("say", help="lloyd posts a message (shorthand for send --from lloyd)")
    say.add_argument("text", help="message body")
    say.add_argument("--reply", help="message id to reply to")
    say.set_defaults(func=cmd_say)

    tail = sub.add_parser("tail", help="show recent messages")
    tail.add_argument("-n", type=int, default=20, help="number of messages")
    tail.set_defaults(func=cmd_tail)

    show = sub.add_parser("show", help="show one message by id")
    show.add_argument("--id", required=True, help="message id")
    show.set_defaults(func=cmd_show)

    lst = sub.add_parser("list", help="list messages")
    lst.add_argument("--from", dest="from_agent", choices=sorted(AGENTS))
    lst.add_argument("--since", help="duration filter, e.g. 2h, 30m, 1d")
    lst.set_defaults(func=cmd_list)

    ids = sub.add_parser("ids", help="list all message ids")
    ids.set_defaults(func=cmd_ids)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
