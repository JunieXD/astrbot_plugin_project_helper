from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import File, Image, Plain, Record, Reply, Video
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.tool import ToolSet
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.platform.message_type import MessageType

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # pragma: no cover - compatibility with older AstrBot versions
    get_astrbot_data_path = None

from .repo_tools import RepositoryTools


PLUGIN_NAME = "astrbot_plugin_project_helper"
ANSWER_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class BufferedMessage:
    sender_id: str
    sender_name: str
    text: str
    outline: str
    attachments: list[str]
    created_at: float


@dataclass
class SessionBuffer:
    messages: list[BufferedMessage] = field(default_factory=list)
    task: asyncio.Task | None = None
    updated_at: float = 0.0


@dataclass(frozen=True)
class ProjectHelperConfig:
    repo_url: str = ""
    repo_branch: str = ""
    repo_path: str = "target_repo"
    enabled_sessions: tuple[str, ...] = ()
    buffer_seconds: float = 8.0
    max_buffer_messages: int = 8
    max_answer_chars: int = 1800
    agent_model: str = ""
    agent_timeout_seconds: float = 180.0
    tool_timeout_seconds: int = 30
    max_tool_calls: int = 10
    conversation_ttl_seconds: float = 300.0
    respond_when_mentioned_only: bool = False
    send_typing: bool = True
    auto_update_repo: bool = False
    include_sources: bool = True
    debug_reply_on_error: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ProjectHelperConfig":
        data = data or {}
        return cls(
            repo_url=str(data.get("repo_url") or "").strip(),
            repo_branch=str(data.get("repo_branch") or "").strip(),
            repo_path=str(data.get("repo_path") or "target_repo").strip() or "target_repo",
            enabled_sessions=tuple(str(item) for item in data.get("enabled_sessions") or [] if str(item).strip()),
            buffer_seconds=_as_float(data.get("buffer_seconds"), 8.0, 0.5, 120.0),
            max_buffer_messages=_as_int(data.get("max_buffer_messages"), 8, 1, 50),
            max_answer_chars=_as_int(data.get("max_answer_chars"), 1800, 200, 8000),
            agent_model=str(data.get("agent_model") or "").strip(),
            agent_timeout_seconds=_as_float(data.get("agent_timeout_seconds"), 180.0, 20.0, 900.0),
            tool_timeout_seconds=_as_int(data.get("tool_timeout_seconds"), 30, 5, 300),
            max_tool_calls=_as_int(data.get("max_tool_calls"), 10, 1, 40),
            conversation_ttl_seconds=_as_float(data.get("conversation_ttl_seconds"), 300.0, 30.0, 3600.0),
            respond_when_mentioned_only=bool(data.get("respond_when_mentioned_only", False)),
            send_typing=bool(data.get("send_typing", True)),
            auto_update_repo=bool(data.get("auto_update_repo", False)),
            include_sources=bool(data.get("include_sources", True)),
            debug_reply_on_error=bool(data.get("debug_reply_on_error", True)),
        )


def _as_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _as_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _plugin_data_dir() -> Path:
    if get_astrbot_data_path is not None:
        return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
    return Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME


@register(
    "astrbot_plugin_project_helper",
    "Junie",
    "Use an AstrBot agent to inspect a GitHub repository and answer project questions in group chat.",
    "0.1.0",
)
class ProjectHelperPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = ProjectHelperConfig.from_mapping(config)
        self.data_dir = _plugin_data_dir()
        self.repo_base_dir = self.data_dir / "repos"
        self.repo_base_dir.mkdir(parents=True, exist_ok=True)
        self.buffers: dict[str, SessionBuffer] = {}
        self._repo_lock = asyncio.Lock()

    @filter.command("project_helper_status")
    async def status(self, event: AstrMessageEvent) -> None:
        repo_root = self._repo_root()
        status = [
            "Project Helper",
            f"repo_path: {repo_root}",
            f"repo_exists: {repo_root.exists()}",
            f"repo_url: {self.config.repo_url or '(local only)'}",
            f"branch: {self.config.repo_branch or '(default)'}",
            f"enabled_sessions: {', '.join(self.config.enabled_sessions) if self.config.enabled_sessions else '(all)'}",
        ]
        event.set_result("\n".join(status))

    @filter.command("project_helper_update")
    async def update_repo(self, event: AstrMessageEvent) -> None:
        if not event.is_admin():
            event.set_result("只有管理员可以更新项目仓库。")
            return
        try:
            repo_root = await self._ensure_repo(update=True)
        except Exception as exc:
            logger.error("Project Helper repository update failed: %s", exc, exc_info=True)
            event.set_result(f"仓库更新失败：{exc}")
            return
        event.set_result(f"仓库已就绪：{repo_root}")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=20)
    async def on_message(self, event: AstrMessageEvent) -> None:
        if not self._should_watch(event):
            return
        if event.get_message_str().lstrip().startswith("/"):
            return
        if self.config.respond_when_mentioned_only and not (
            event.is_wake_up() or event.is_at_or_wake_command
        ):
            return
        event.should_call_llm(True)

        message = await self._build_buffered_message(event)
        if not message.text and not message.attachments:
            return

        session_id = event.unified_msg_origin
        buf = self.buffers.setdefault(session_id, SessionBuffer())
        buf.messages.append(message)
        buf.updated_at = time.time()
        if len(buf.messages) > self.config.max_buffer_messages:
            buf.messages = buf.messages[-self.config.max_buffer_messages :]

        if buf.task and not buf.task.done():
            buf.task.cancel()
        buf.task = asyncio.create_task(self._delayed_process(event, session_id))

    async def _delayed_process(self, event: AstrMessageEvent, session_id: str) -> None:
        try:
            await asyncio.sleep(self.config.buffer_seconds)
            await self._process_buffer(event, session_id)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Project Helper processing failed: %s", exc, exc_info=True)
            if self.config.debug_reply_on_error:
                await event.send(MessageChain([Plain(f"项目问答处理失败：{exc}")]))

    async def _process_buffer(self, event: AstrMessageEvent, session_id: str) -> None:
        buf = self.buffers.get(session_id)
        if not buf or not buf.messages:
            return

        now = time.time()
        messages = [
            item
            for item in buf.messages
            if now - item.created_at <= self.config.conversation_ttl_seconds
        ]
        self.buffers.pop(session_id, None)
        if not messages:
            return

        if self.config.send_typing:
            await event.send_typing()
        try:
            repo_root = await self._ensure_repo(update=self.config.auto_update_repo)
            result = await self._ask_agent(event, repo_root, messages)
        finally:
            if self.config.send_typing:
                await event.stop_typing()

        if not result.get("reply"):
            return

        answer = str(result.get("answer") or "").strip()
        if not answer:
            return
        if len(answer) > self.config.max_answer_chars:
            answer = answer[: self.config.max_answer_chars].rstrip() + "\n...后面略了，我先把关键结论放上面。"
        await event.send(MessageChain([Plain(answer)]))

    async def _ask_agent(
        self,
        event: AstrMessageEvent,
        repo_root: Path,
        messages: list[BufferedMessage],
    ) -> dict[str, Any]:
        provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
        repo_tools = RepositoryTools(repo_root)
        tool_set = ToolSet()
        for tool in repo_tools.tool_set():
            tool_set.add_tool(tool)

        prompt = self._build_user_prompt(event, repo_root, messages)
        system_prompt = self._build_system_prompt()

        kwargs: dict[str, Any] = {
            "event": event,
            "chat_provider_id": provider_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "tools": tool_set,
            "contexts": [],
            "max_steps": self.config.max_tool_calls,
            "tool_call_timeout": self.config.tool_timeout_seconds,
            "stream": False,
        }
        response = await asyncio.wait_for(
            self.context.tool_loop_agent(**kwargs),
            timeout=self.config.agent_timeout_seconds,
        )
        raw = (response.completion_text or "").strip()
        parsed = self._parse_agent_json(raw)
        if parsed is None:
            logger.warning("Project Helper agent returned non-JSON response: %s", raw)
            return {"reply": True, "answer": raw}
        return parsed

    def _build_system_prompt(self) -> str:
        sources_rule = (
            "回答中尽量附上关键文件路径和行号，比如 `src/foo.py:123`。"
            if self.config.include_sources
            else "回答中不必强制列出文件路径。"
        )
        return (
            "你是一个 GitHub 项目交流群里的技术答疑成员。你的任务不是闲聊，而是判断群友最近几条消息是否在问目标项目。"
            "你可以使用仓库只读工具查看目录、搜索代码、读取文件和 Markdown。"
            "如果问题需要代码依据，必须先用工具调查；不要凭空猜。"
            "如果最近消息只是闲聊、寒暄、表情、无关话题，返回不回复。"
            "如果信息不足但明显与项目有关，可以基于已知代码说明你确认了什么、还缺什么。"
            "语气自然、像群友，直接给结论，少说流程。"
            f"{sources_rule}"
            "最终只能输出一个 JSON 对象，不要输出 Markdown 代码块："
            "{\"reply\": boolean, \"answer\": string, \"confidence\": \"low|medium|high\"}。"
            "当 reply=false 时 answer 必须为空字符串。"
        )

    def _build_user_prompt(
        self,
        event: AstrMessageEvent,
        repo_root: Path,
        messages: list[BufferedMessage],
    ) -> str:
        lines = [
            f"目标仓库本地路径：{repo_root}",
            f"平台会话：{event.unified_msg_origin}",
            "最近连续消息：",
        ]
        for idx, item in enumerate(messages, start=1):
            attachments = f" 附件/媒体: {', '.join(item.attachments)}" if item.attachments else ""
            text = item.text or item.outline or "(无文本)"
            lines.append(f"{idx}. {item.sender_name or item.sender_id}: {text}{attachments}")
        lines.append(
            "请判断这些消息合起来是否是在问目标项目。如果是，调查仓库并回答；如果不是，返回 reply=false。"
        )
        return "\n".join(lines)

    def _parse_agent_json(self, raw: str) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = ANSWER_JSON_RE.search(raw)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict):
            return None
        return {
            "reply": bool(data.get("reply", False)),
            "answer": str(data.get("answer") or "").strip(),
            "confidence": str(data.get("confidence") or "medium"),
        }

    def _should_watch(self, event: AstrMessageEvent) -> bool:
        if self.config.enabled_sessions and event.unified_msg_origin not in self.config.enabled_sessions:
            return False
        if event.get_message_type() not in {
            MessageType.GROUP_MESSAGE,
            MessageType.FRIEND_MESSAGE,
        }:
            return False
        if event.get_sender_id() == event.get_self_id():
            return False
        return True

    async def _build_buffered_message(self, event: AstrMessageEvent) -> BufferedMessage:
        attachments = []
        for comp in event.get_messages():
            desc = await self._describe_component(comp)
            if desc:
                attachments.append(desc)
        return BufferedMessage(
            sender_id=event.get_sender_id(),
            sender_name=event.get_sender_name(),
            text=event.get_message_str().strip(),
            outline=event.get_message_outline().strip(),
            attachments=attachments,
            created_at=time.time(),
        )

    async def _describe_component(self, comp: BaseMessageComponent) -> str:
        if isinstance(comp, Plain):
            return ""
        if isinstance(comp, Image):
            return self._media_desc("图片", comp)
        if isinstance(comp, File):
            name = getattr(comp, "name", "") or getattr(comp, "file", "") or "文件"
            return f"文件({name})"
        if isinstance(comp, Record):
            return self._media_desc("语音", comp)
        if isinstance(comp, Video):
            return self._media_desc("视频", comp)
        if isinstance(comp, Reply):
            return f"引用消息({getattr(comp, 'sender_nickname', '')}: {getattr(comp, 'message_str', '')})"
        comp_type = getattr(comp, "type", type(comp).__name__)
        return f"{comp_type}"

    def _media_desc(self, label: str, comp: BaseMessageComponent) -> str:
        for attr in ("path", "file", "url"):
            value = getattr(comp, attr, None)
            if value:
                text = str(value)
                if text.startswith("base64://"):
                    return f"{label}(base64)"
                if len(text) > 160:
                    text = text[:157] + "..."
                return f"{label}({text})"
        return label

    def _repo_root(self) -> Path:
        configured = Path(self.config.repo_path).expanduser()
        if configured.is_absolute():
            return configured.resolve()
        return (self.repo_base_dir / configured).resolve()

    async def _ensure_repo(self, *, update: bool) -> Path:
        async with self._repo_lock:
            repo_root = self._repo_root()
            if repo_root.exists() and (repo_root / ".git").exists():
                if update:
                    await asyncio.to_thread(self._git_update, repo_root)
                return repo_root

            if repo_root.exists() and not (repo_root / ".git").exists():
                if self.config.repo_url:
                    raise RuntimeError(f"repo_path exists but is not a git repository: {repo_root}")
                return repo_root

            if not self.config.repo_url:
                raise RuntimeError("repo_url 未配置，且 repo_path 不存在。请先配置目标仓库或运行本地 checkout。")

            repo_root.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._git_clone, repo_root)
            return repo_root

    def _git_clone(self, repo_root: Path) -> None:
        if shutil.which("git") is None:
            raise RuntimeError("系统找不到 git，无法克隆仓库。")
        cmd = ["git", "clone", "--depth", "1"]
        if self.config.repo_branch:
            cmd.extend(["--branch", self.config.repo_branch])
        cmd.extend([self.config.repo_url, str(repo_root)])
        self._run_cmd(cmd, cwd=self.repo_base_dir)

    def _git_update(self, repo_root: Path) -> None:
        if shutil.which("git") is None:
            raise RuntimeError("系统找不到 git，无法更新仓库。")
        self._run_cmd(["git", "fetch", "--all", "--prune"], cwd=repo_root)
        if self.config.repo_branch:
            self._run_cmd(["git", "checkout", self.config.repo_branch], cwd=repo_root)
        self._run_cmd(["git", "pull", "--ff-only"], cwd=repo_root)

    def _run_cmd(self, cmd: list[str], cwd: Path) -> None:
        env = os.environ.copy()
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(stderr[:1000] or f"command failed: {' '.join(cmd)}")

    async def terminate(self) -> None:
        for buf in self.buffers.values():
            if buf.task and not buf.task.done():
                buf.task.cancel()
        self.buffers.clear()
