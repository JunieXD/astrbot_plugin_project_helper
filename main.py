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
from typing import Any, Awaitable, Callable

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.message_components import File, Image, Plain, Record, Reply, Video
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.tool import FunctionTool, ToolSet
from astrbot.core.message.components import BaseMessageComponent
from astrbot.core.platform.message_type import MessageType

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_data_path
except Exception:  # pragma: no cover - compatibility with older AstrBot versions
    get_astrbot_data_path = None

from .repo_tools import QAMemoryTools, RepositoryTools


PLUGIN_NAME = "astrbot_plugin_project_helper"
ANSWER_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
RECENT_ANSWER_TTL_SECONDS = 3600.0
DEFAULT_ANSWER_STYLE_PROMPT = (
    "像项目群里一个熟悉项目的真人群友在顺手回复。"
    "默认 1 到 4 句话，除非用户明确要教程，不要写长篇编号列表。"
    "先给结论或下一步操作，再补一句原因；语气轻松、直接、有边界，不要客服腔。"
    "可以说“先看这个”“一般是这样”“这个不用重抓整站”，不要说“建议您按照以下步骤”。"
    "示例好回复：先看缺邮箱候选有没有个人主页链接。有的话在抓取审核页选中它，点补全导师资料，系统会进详情页继续找邮箱。"
    "示例差回复：好的，这个问题明显是项目相关的，我来回答。常见原因如下：1...2...3..."
)


@dataclass(frozen=True)
class ProjectBinding:
    group_id: str
    repo_url: str = ""
    repo_branch: str = ""
    repo_path: str = ""
    qa_path: str = ""
    project_prompt: str = ""
    enabled: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ProjectBinding":
        data = data or {}
        legacy_session = str(data.get("session_id") or data.get("target_umo") or "").strip()
        group_id = _extract_group_id(data.get("group_id") or data.get("qq_group_id"))
        if not group_id:
            group_id = _extract_group_id(legacy_session)
        return cls(
            group_id=group_id,
            repo_url=str(data.get("repo_url") or "").strip(),
            repo_branch=str(data.get("repo_branch") or "").strip(),
            repo_path=str(data.get("repo_path") or "").strip(),
            qa_path=str(data.get("qa_path") or "").strip(),
            project_prompt=str(data.get("project_prompt") or "").strip(),
            enabled=_as_bool(data.get("enabled"), True),
        )

    def label(self) -> str:
        return _repo_name_from_url(self.repo_url) or self.repo_path or self.group_id or "(未命名项目)"


@dataclass
class BufferedMessage:
    sender_id: str
    sender_name: str
    is_admin: bool
    text: str
    outline: str
    attachments: list[str]
    created_at: float


@dataclass
class SessionBuffer:
    messages: list[BufferedMessage] = field(default_factory=list)
    task: asyncio.Task | None = None
    updated_at: float = 0.0
    truncated_count: int = 0
    processing: bool = False
    running_messages: list[BufferedMessage] = field(default_factory=list)
    running_truncated_count: int = 0


@dataclass
class RecentAnswer:
    project_key: str
    answer: str
    question_context: str
    created_at: float


class AgentTraceRecorder:
    def __init__(self, *, run_id: str, max_tool_calls: int) -> None:
        self.run_id = run_id
        self.max_tool_calls = max_tool_calls
        self.started_at = time.time()
        self.finished_at = 0.0
        self.tool_calls: list[dict[str, Any]] = []
        self.tool_counts: dict[str, int] = {}
        self.qa_upsert_called = False
        self.system_prompt = ""
        self.user_prompt = ""
        self.raw_response = ""
        self.parsed_response: dict[str, Any] | None = None
        self.post_check: dict[str, Any] | None = None
        self.final_state: dict[str, Any] = {}

    @property
    def used_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.max_tool_calls - self.used_tool_calls)

    async def run_tool(
        self,
        name: str,
        args: dict[str, object],
        call: Callable[[], Awaitable[str]],
    ) -> str:
        index = self.used_tool_calls + 1
        if self.used_tool_calls >= self.max_tool_calls:
            result = (
                f"TOOL_ERROR {name}: tool budget exhausted. "
                "Return the final JSON now unless the answer is impossible."
            )
            self._record_tool_call(
                index=index,
                name=name,
                args=args,
                ok=False,
                duration_ms=0,
                result=result,
                error="tool budget exhausted",
            )
            return self._with_budget_hint(result)

        started = time.monotonic()
        ok = True
        error = ""
        try:
            result = await call()
        except Exception as exc:
            ok = False
            error = f"{type(exc).__name__}: {exc}"
            result = f"TOOL_ERROR {name}: {error}"

        duration_ms = int((time.monotonic() - started) * 1000)
        if name == "qa_upsert" and ok:
            self.qa_upsert_called = True
        self._record_tool_call(
            index=index,
            name=name,
            args=args,
            ok=ok,
            duration_ms=duration_ms,
            result=result,
            error=error,
        )
        return self._with_budget_hint(result)

    def finish(
        self,
        *,
        parsed_response: dict[str, Any] | None,
        raw_response: str,
        final_state: dict[str, Any],
    ) -> None:
        self.finished_at = time.time()
        self.parsed_response = parsed_response
        self.raw_response = raw_response
        self.final_state = final_state

    def to_payload(
        self,
        *,
        event: AstrMessageEvent,
        project: ProjectBinding,
        repo_root: Path,
        qa_path: Path,
        trigger_messages: list[BufferedMessage],
        trigger_truncated_count: int,
        followup_messages: list[BufferedMessage],
        followup_truncated_count: int,
    ) -> dict[str, Any]:
        finished_at = self.finished_at or time.time()
        return {
            "run_id": self.run_id,
            "started_at": _format_ts(self.started_at),
            "finished_at": _format_ts(finished_at),
            "duration_ms": int((finished_at - self.started_at) * 1000),
            "group_id": project.group_id,
            "session_id": event.unified_msg_origin,
            "project": project.label(),
            "repo_root": str(repo_root),
            "qa_path": str(qa_path),
            "tool_budget": {
                "max": self.max_tool_calls,
                "used": self.used_tool_calls,
                "remaining": self.remaining_tool_calls,
                "counts": self.tool_counts,
                "qa_upsert_called": self.qa_upsert_called,
            },
            "trigger_messages": _messages_to_trace(trigger_messages),
            "trigger_truncated_count": trigger_truncated_count,
            "followup_messages": _messages_to_trace(followup_messages),
            "followup_truncated_count": followup_truncated_count,
            "prompts": {
                "system": _clip_text(self.system_prompt, 20000),
                "user": _clip_text(self.user_prompt, 20000),
            },
            "tools": self.tool_calls,
            "agent": {
                "raw_response": _clip_text(self.raw_response, 20000),
                "parsed_response": self.parsed_response,
            },
            "post_check": self.post_check,
            "final": self.final_state,
        }

    def _record_tool_call(
        self,
        *,
        index: int,
        name: str,
        args: dict[str, object],
        ok: bool,
        duration_ms: int,
        result: str,
        error: str,
    ) -> None:
        self.tool_counts[name] = self.tool_counts.get(name, 0) + 1
        self.tool_calls.append(
            {
                "index": index,
                "name": name,
                "args": _sanitize_trace_value(args),
                "ok": ok,
                "duration_ms": duration_ms,
                "remaining_budget": max(0, self.max_tool_calls - index),
                "result_preview": _clip_text(str(result), 1200),
                "error": error,
            }
        )

    def _with_budget_hint(self, result: str) -> str:
        hint = (
            f"[Project Helper tool budget: used {self.used_tool_calls}/{self.max_tool_calls}, "
            f"remaining {self.remaining_tool_calls}, "
            f"qa_upsert_called={str(self.qa_upsert_called).lower()}]"
        )
        if self.remaining_tool_calls <= 1:
            hint += " If the answer is ready, return the final JSON now; reserve the last call for qa_upsert only when the conclusion is reusable and confirmed."
        return f"{result}\n\n{hint}"


@dataclass(frozen=True)
class ProjectHelperConfig:
    projects: tuple[ProjectBinding, ...] = ()
    admin_qqs: tuple[str, ...] = ()
    buffer_seconds: float = 15.0
    max_buffer_messages: int = 20
    max_answer_chars: int = 700
    answer_style_prompt: str = DEFAULT_ANSWER_STYLE_PROMPT
    agent_timeout_seconds: float = 300.0
    tool_timeout_seconds: int = 30
    max_tool_calls: int = 25
    conversation_ttl_seconds: float = 300.0
    respond_when_mentioned_only: bool = False
    send_typing: bool = True
    auto_update_repo: bool = True
    include_sources: bool = False
    error_notify_admins: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ProjectHelperConfig":
        data = data or {}
        projects = cls._parse_projects(data)
        return cls(
            projects=projects,
            admin_qqs=_as_string_tuple(data.get("admin_qqs") or data.get("admin_qq_ids")),
            buffer_seconds=_as_float(data.get("buffer_seconds"), 15.0, 1.0, 180.0),
            max_buffer_messages=_as_int(data.get("max_buffer_messages"), 20, 1, 100),
            max_answer_chars=_as_int(data.get("max_answer_chars"), 700, 200, 8000),
            answer_style_prompt=str(
                data.get("answer_style_prompt") or DEFAULT_ANSWER_STYLE_PROMPT
            ).strip(),
            agent_timeout_seconds=_as_float(data.get("agent_timeout_seconds"), 300.0, 20.0, 1200.0),
            tool_timeout_seconds=_as_int(data.get("tool_timeout_seconds"), 30, 5, 300),
            max_tool_calls=_as_int(data.get("max_tool_calls"), 25, 1, 80),
            conversation_ttl_seconds=_as_float(data.get("conversation_ttl_seconds"), 300.0, 30.0, 3600.0),
            respond_when_mentioned_only=_as_bool(data.get("respond_when_mentioned_only"), False),
            send_typing=_as_bool(data.get("send_typing"), True),
            auto_update_repo=_as_bool(data.get("auto_update_repo"), True),
            include_sources=_as_bool(data.get("include_sources"), False),
            error_notify_admins=_as_bool(
                data.get("error_notify_admins", data.get("debug_reply_on_error")),
                True,
            ),
        )

    @staticmethod
    def _parse_projects(data: dict[str, Any]) -> tuple[ProjectBinding, ...]:
        raw_projects = data.get("projects")
        projects: list[ProjectBinding] = []
        if isinstance(raw_projects, list):
            for item in raw_projects:
                if not isinstance(item, dict):
                    continue
                project = ProjectBinding.from_mapping(item)
                if project.group_id:
                    projects.append(project)

        if projects:
            return tuple(projects)

        legacy_repo_url = str(data.get("repo_url") or "").strip()
        legacy_repo_path = str(data.get("repo_path") or "").strip()
        if legacy_repo_url:
            sessions = [
                str(item).strip()
                for item in data.get("enabled_sessions") or []
                if str(item).strip()
            ]
            return tuple(
                ProjectBinding(
                    group_id=group_id,
                    repo_url=legacy_repo_url,
                    repo_branch=str(data.get("repo_branch") or "").strip(),
                    repo_path=legacy_repo_path,
                    project_prompt=str(data.get("project_prompt") or "").strip(),
                    enabled=True,
                )
                for session in sessions
                if (group_id := _extract_group_id(session))
            )
        return ()


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


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_items = re.split(r"[\s,，;；]+", value)
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]
    items = [item.strip() for item in raw_items if item and item.strip()]
    return tuple(dict.fromkeys(items))


def _extract_group_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if ":" in text:
        parts = text.split(":")
        if len(parts) >= 3 and parts[-2] == MessageType.GROUP_MESSAGE.value:
            return parts[-1].strip()
        return ""
    return text


def _repo_name_from_url(repo_url: str) -> str:
    text = str(repo_url or "").strip().removesuffix(".git").rstrip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].strip() or text


def _plugin_data_dir() -> Path:
    if get_astrbot_data_path is not None:
        return Path(get_astrbot_data_path()) / "plugin_data" / PLUGIN_NAME
    return Path.cwd() / "data" / "plugin_data" / PLUGIN_NAME


def _format_ts(value: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value or time.time()))


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n... clipped at {limit} chars"


def _sanitize_trace_value(value: Any, *, limit: int = 1200) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_trace_value(item, limit=limit)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_trace_value(item, limit=limit) for item in value[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return _clip_text(value, limit)
        return value
    return _clip_text(str(value), limit)


def _messages_to_trace(messages: list[BufferedMessage]) -> list[dict[str, Any]]:
    return [
        {
            "sender_id": item.sender_id,
            "sender_name": item.sender_name,
            "is_admin": item.is_admin,
            "text": _clip_text(item.text, 2000),
            "outline": _clip_text(item.outline, 1000),
            "attachments": item.attachments,
            "created_at": _format_ts(item.created_at),
        }
        for item in messages
    ]


@register(
    "astrbot_plugin_project_helper",
    "JunieXD",
    "Use an AstrBot agent to inspect a GitHub repository and answer project questions in group chat.",
    "0.4.0",
)
class ProjectHelperPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any] | None = None) -> None:
        super().__init__(context)
        self.config = ProjectHelperConfig.from_mapping(config)
        self.data_dir = _plugin_data_dir()
        self.repo_base_dir = self.data_dir / "repos"
        self.trace_base_dir = self.data_dir / "traces"
        self.repo_base_dir.mkdir(parents=True, exist_ok=True)
        self.trace_base_dir.mkdir(parents=True, exist_ok=True)
        self.buffers: dict[str, SessionBuffer] = {}
        self.recent_answers: dict[str, RecentAnswer] = {}
        self._repo_locks: dict[str, asyncio.Lock] = {}

    @filter.command_group("ph")
    def ph(self) -> None:
        pass

    @ph.command("status")
    async def status(self, event: AstrMessageEvent) -> None:
        if not self._is_admin(event):
            event.set_result("只有管理员可以查看项目答疑助手状态。")
            return
        project = self._project_for_event(event)
        if project is None:
            event.set_result(
                f"当前群未绑定项目：{event.get_group_id() or '(非群聊)'}\n"
                "请在插件配置 projects 中添加一条 QQ群号 为当前群号、Git 仓库地址非空的项目配置。"
            )
            return
        if not project.repo_url:
            event.set_result(
                f"当前群已配置项目绑定，但 Git 仓库地址为空：{project.group_id}\n"
                "请在插件配置 projects 中填写 Git 仓库地址。"
            )
            return
        repo_root = self._repo_root(project)
        qa_path = self._qa_path(project)
        status = [
            "Project Helper (/ph)",
            f"group_id: {project.group_id}",
            f"project: {project.label()}",
            f"repo_path: {repo_root}",
            f"repo_exists: {repo_root.exists()}",
            f"repo_url: {project.repo_url}",
            f"branch: {project.repo_branch or '(auto: main -> master -> remote default)'}",
            f"qa_path: {qa_path}",
            f"qa_exists: {qa_path.exists()}",
            f"trace_path: {self._latest_trace_path(project)}",
            f"trace_exists: {self._latest_trace_path(project).exists()}",
            f"max_tool_calls: {self.config.max_tool_calls}",
            f"buffer_seconds: {self.config.buffer_seconds:g}",
            f"max_buffer_messages: {self.config.max_buffer_messages}",
        ]
        event.set_result("\n".join(status))

    @ph.command("trace")
    async def trace(self, event: AstrMessageEvent) -> None:
        if not self._is_admin(event):
            event.set_result("只有管理员可以查看项目答疑助手 trace。")
            return
        project = self._project_for_event(event)
        if project is None:
            event.set_result(f"当前群未绑定项目：{event.get_group_id() or '(非群聊)'}")
            return
        trace_path = self._latest_trace_path(project)
        if not trace_path.exists():
            event.set_result(f"当前群还没有 Agent trace。\ntrace_path: {trace_path}")
            return
        try:
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
        except Exception as exc:
            event.set_result(f"读取 trace 失败：{exc}\ntrace_path: {trace_path}")
            return

        tool_budget = payload.get("tool_budget") or {}
        final = payload.get("final") or {}
        agent = payload.get("agent") or {}
        post_check = payload.get("post_check") or {}
        parsed = agent.get("parsed_response") or {}
        answer = str(parsed.get("answer") or "").strip()
        lines = [
            "Project Helper trace",
            f"run_id: {payload.get('run_id', '')}",
            f"time: {payload.get('started_at', '')} -> {payload.get('finished_at', '')}",
            f"project: {payload.get('project', project.label())}",
            f"reply: {parsed.get('reply')}",
            f"confidence: {parsed.get('confidence', '')}",
            f"reason: {_clip_text(str(parsed.get('reason') or ''), 180)}",
            f"tool_calls: {tool_budget.get('used', 0)}/{tool_budget.get('max', self.config.max_tool_calls)}",
            f"tool_counts: {json.dumps(tool_budget.get('counts', {}), ensure_ascii=False)}",
            f"qa_upsert_called: {tool_budget.get('qa_upsert_called', False)}",
            f"suppressed: {final.get('suppressed', False)}",
        ]
        if post_check:
            lines.append(f"post_check: {post_check.get('decision', '')} {_clip_text(str(post_check.get('reason') or ''), 160)}")
        if answer:
            lines.append(f"answer: {_clip_text(answer, 260)}")
        lines.append(f"trace_path: {trace_path}")
        event.set_result("\n".join(lines))

    @ph.command("update")
    async def update_repo(self, event: AstrMessageEvent) -> None:
        if not self._is_admin(event):
            event.set_result("只有管理员可以更新项目仓库。")
            return
        project = self._project_for_event(event)
        if project is None:
            event.set_result(f"当前群未绑定项目：{event.get_group_id() or '(非群聊)'}")
            return
        if not project.repo_url:
            event.set_result("当前群项目绑定缺少 Git 仓库地址，请先在插件配置里填写。")
            return
        try:
            repo_root = await self._ensure_repo(project, update=True)
        except Exception as exc:
            logger.error("Project Helper repository update failed: %s", exc, exc_info=True)
            event.set_result(f"仓库更新失败：{exc}")
            return
        event.set_result(f"仓库已就绪：{project.label()}\n{repo_root}")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=20)
    async def on_message(self, event: AstrMessageEvent) -> None:
        project = self._project_for_event(event)
        if not self._should_watch(event, project):
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
        buf.updated_at = time.time()
        if buf.processing:
            self._append_buffer_message(buf.running_messages, message, running=True, buf=buf)
            return

        self._append_buffer_message(buf.messages, message, running=False, buf=buf)

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
            await self._notify_admins(event, f"项目问答处理失败：{exc}")

    async def _process_buffer(self, event: AstrMessageEvent, session_id: str) -> None:
        buf = self.buffers.get(session_id)
        if not buf or not buf.messages:
            return
        project = self._project_for_event(event)
        if project is None:
            self.buffers.pop(session_id, None)
            return

        now = time.time()
        messages = [
            item
            for item in buf.messages
            if now - item.created_at <= self.config.conversation_ttl_seconds
        ]
        truncated_count = buf.truncated_count
        buf.messages = []
        buf.truncated_count = 0
        buf.running_messages = []
        buf.running_truncated_count = 0
        buf.processing = True
        if not messages:
            buf.processing = False
            return

        trace: AgentTraceRecorder | None = None
        result: dict[str, Any] = {}
        repo_root: Path | None = None
        processing_error: Exception | None = None
        if self.config.send_typing:
            await event.send_typing()
        try:
            repo_root = await self._ensure_repo(project, update=self.config.auto_update_repo)
            trace = AgentTraceRecorder(
                run_id=f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}",
                max_tool_calls=self.config.max_tool_calls,
            )
            result = await self._ask_agent(event, project, repo_root, messages, truncated_count, trace)
        except Exception as exc:
            processing_error = exc
        finally:
            if self.config.send_typing:
                await event.stop_typing()
            buf.processing = False

        followup_messages = [
            item
            for item in buf.running_messages
            if time.time() - item.created_at <= self.config.conversation_ttl_seconds
        ]
        followup_truncated_count = buf.running_truncated_count
        buf.running_messages = []
        buf.running_truncated_count = 0
        replay_followups = False

        if processing_error is not None:
            if trace and repo_root is not None:
                trace.post_check = {"decision": "skipped", "reason": "Agent run failed before reply."}
                trace.finish(
                    parsed_response=None,
                    raw_response=trace.raw_response,
                    final_state={
                        "reply_sent": False,
                        "suppressed": False,
                        "reason": f"error: {type(processing_error).__name__}: {processing_error}",
                    },
                )
                self._write_trace(
                    event,
                    project,
                    repo_root,
                    trace,
                    messages,
                    truncated_count,
                    followup_messages,
                    followup_truncated_count,
                )
            self._drop_idle_buffer(
                event,
                session_id,
                buf,
                replay_messages=followup_messages if followup_messages else None,
                replay_truncated_count=followup_truncated_count,
            )
            raise processing_error

        if not result.get("reply"):
            final_state = {"reply_sent": False, "suppressed": False, "reason": "agent_reply_false"}
            if trace and repo_root is not None:
                trace.finish(parsed_response=result, raw_response=trace.raw_response, final_state=final_state)
                self._write_trace(event, project, repo_root, trace, messages, truncated_count, followup_messages, followup_truncated_count)
            replay_followups = self._should_replay_followups(followup_messages)
            self._drop_idle_buffer(event, session_id, buf, replay_messages=followup_messages if replay_followups else None, replay_truncated_count=followup_truncated_count)
            return

        answer = str(result.get("answer") or "").strip()
        if not answer:
            final_state = {"reply_sent": False, "suppressed": False, "reason": "empty_answer"}
            if trace and repo_root is not None:
                trace.finish(parsed_response=result, raw_response=trace.raw_response, final_state=final_state)
                self._write_trace(event, project, repo_root, trace, messages, truncated_count, followup_messages, followup_truncated_count)
            replay_followups = self._should_replay_followups(followup_messages)
            self._drop_idle_buffer(event, session_id, buf, replay_messages=followup_messages if replay_followups else None, replay_truncated_count=followup_truncated_count)
            return
        suppressed = await self._should_suppress_after_followup(
            event,
            project,
            messages,
            answer,
            followup_messages,
            trace,
        )
        if suppressed:
            final_state = {"reply_sent": False, "suppressed": True, "reason": "followup_solved_or_superseded"}
            if trace and repo_root is not None:
                trace.finish(parsed_response=result, raw_response=trace.raw_response, final_state=final_state)
                self._write_trace(event, project, repo_root, trace, messages, truncated_count, followup_messages, followup_truncated_count)
            self._drop_idle_buffer(event, session_id, buf)
            return
        if len(answer) > self.config.max_answer_chars:
            answer = answer[: self.config.max_answer_chars].rstrip() + "\n...后面略了，我先把关键结论放上面。"
        await event.send(MessageChain([Plain(answer)]))
        self._remember_answer(event, project, messages, answer)
        final_state = {"reply_sent": True, "suppressed": False, "reason": "sent"}
        if trace and repo_root is not None:
            trace.finish(parsed_response=result, raw_response=trace.raw_response, final_state=final_state)
            self._write_trace(event, project, repo_root, trace, messages, truncated_count, followup_messages, followup_truncated_count)
        replay_followups = self._should_replay_followups(followup_messages)
        self._drop_idle_buffer(event, session_id, buf, replay_messages=followup_messages if replay_followups else None, replay_truncated_count=followup_truncated_count)

    async def _ask_agent(
        self,
        event: AstrMessageEvent,
        project: ProjectBinding,
        repo_root: Path,
        messages: list[BufferedMessage],
        truncated_count: int,
        trace: AgentTraceRecorder,
    ) -> dict[str, Any]:
        repo_tools = RepositoryTools(repo_root)
        qa_tools = QAMemoryTools(self._qa_path(project), project_label=project.label())
        tool_set = ToolSet()
        for tool in [*qa_tools.tool_set(), *repo_tools.tool_set()]:
            tool_set.add_tool(self._wrap_tool(tool, trace))

        prompt = self._build_user_prompt(event, project, repo_root, messages, truncated_count)
        system_prompt = self._build_system_prompt()
        trace.user_prompt = prompt
        trace.system_prompt = system_prompt
        provider_id = await self._provider_id_for_event(event)

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
        trace.raw_response = raw
        parsed = self._parse_agent_json(raw)
        if parsed is None:
            logger.warning("Project Helper agent returned non-JSON response: %s", raw)
            parsed = {"reply": True, "answer": raw, "confidence": "low", "reason": "non_json_agent_response"}
        trace.parsed_response = parsed
        return parsed

    def _wrap_tool(self, tool: FunctionTool, trace: AgentTraceRecorder) -> FunctionTool:
        name = str(self._tool_attr(tool, "name", "unknown_tool"))
        description = str(self._tool_attr(tool, "description", ""))
        parameters = self._tool_attr(tool, "parameters", {"type": "object", "properties": {}})
        handler = self._tool_attr(tool, "handler", None)
        if handler is None:
            raise RuntimeError(f"Tool {name} has no handler.")

        async def traced_handler(**kwargs: object) -> str:
            async def call() -> str:
                result = handler(**kwargs)
                if hasattr(result, "__await__"):
                    result = await result
                return str(result)

            return await trace.run_tool(name, dict(kwargs), call)

        return FunctionTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=traced_handler,
        )

    def _tool_attr(self, tool: FunctionTool, name: str, default: Any = None) -> Any:
        if hasattr(tool, name):
            return getattr(tool, name)
        data = getattr(tool, "__dict__", {})
        if isinstance(data, dict) and name in data:
            return data[name]
        return default

    def _build_system_prompt(self) -> str:
        sources_rule = (
            "如果做过代码或文档调查，回答末尾用很短的括号补充关键依据，比如 `src/foo.py:123`。"
            if self.config.include_sources
            else "回答中不必强制列出文件路径。"
        )
        return (
            "你是一个 GitHub 项目交流群里的答疑成员，服务对象主要是普通项目使用者，不是代码贡献者。"
            "回答时优先从产品行为、配置方法、使用步骤、报错含义和排查顺序解释；只有用户明显需要实现细节时才展开代码。"
            "你的第一步是判断最近连续消息是否需要机器人回复：如果只是闲聊、寒暄、表情、无关话题，返回不回复。"
            "如果群里后续消息已经把问题回答清楚，或者提问者表示已解决，也返回不回复。"
            "如果无法仅凭聊天内容判断是否属于目标项目问题，可以先用 QA 和仓库工具做少量调查，再决定 reply=true 或 reply=false；调查不代表必须回复。"
            "如果管理员指出你之前的回答有误，必须重新检查 QA 记录和仓库；确认你错了就道歉并给出更正，随后更新 QA；"
            "如果管理员说法不对，就礼貌纠正并给出简短证据。"
            "你有一个项目 QA Markdown 记忆工具。开始调查代码前，优先读取/搜索 QA，看是否已有可靠答案；"
            "若 QA 已足够，直接基于 QA 回答。若 QA 不足或可能过期，再使用仓库只读工具查看目录、搜索代码、读取文件和 Markdown。"
            "如果调查得到的结论可复用、已确认、以后群友可能还会问，应该先用 qa_upsert 把问题、简短答案、适用条件和关键依据写回 QA，再给最终 JSON 回复。"
            "不要为了闲聊、临时猜测、低置信度结论、已被群友解答的问题写 QA；QA 不是聊天流水账。"
            "如果 QA 文件不存在，qa_read/qa_search 返回空或不存在是正常情况；不要反复读取 QA，直接调查仓库，需要沉淀时调用一次 qa_upsert 即可。"
            "每次工具结果末尾会附带当前工具调用预算和 qa_upsert 是否已调用；你能据此控制调查范围。"
            "如果你决定不写 QA，也没关系，但要在最终 JSON 的 reason 里简短说明是临时问题、低置信度、已被群友解答，还是不够通用。"
            "仓库工具只读；不要要求用户理解代码实现，除非这是解决问题必须的信息。"
            "语气自然、像群友，直接给结论，少说流程。"
            "不要说“这个问题明显是项目相关的”“我来回答”“我去查一下”这类暴露判断过程或机器人身份的开场白。"
            "不要把回答写成泛泛的客服排查清单；先给项目内真实存在的页面、按钮、状态、限制和推荐操作，再补充必要的排查项。"
            "如果仓库里能查到具体机制，就不要凭通用经验猜测；不确定的原因要明确说“可能”，不要编造项目能力或限制。"
            "用户问“怎么办”时，优先回答下一步该怎么操作；最后只问一个最关键的补充信息，不要连续追问多个问题。"
            f"回复风格要求：{self.config.answer_style_prompt}"
            f"{sources_rule}"
            "最终只能输出一个 JSON 对象，不要输出 Markdown 代码块："
            "{\"reply\": boolean, \"answer\": string, \"confidence\": \"low|medium|high\", \"reason\": string}。"
            "当 reply=false 时 answer 必须为空字符串。"
        )

    def _build_user_prompt(
        self,
        event: AstrMessageEvent,
        project: ProjectBinding,
        repo_root: Path,
        messages: list[BufferedMessage],
        truncated_count: int,
    ) -> str:
        lines = [
            f"目标项目：{project.label()}",
            f"目标仓库本地路径：{repo_root}",
            f"QA Markdown 路径：{self._qa_path(project)}",
            f"平台会话：{event.unified_msg_origin}",
            f"本轮最多可调用工具 {self.config.max_tool_calls} 次；工具结果会提示已用/剩余次数。请控制调查范围，先查 QA，再用少量仓库搜索定位答案。若结论需要沉淀，请在最终 JSON 前预留一次 qa_upsert。",
        ]
        if project.project_prompt:
            lines.extend(
                [
                    "项目简介/判断边界：",
                    project.project_prompt,
                ]
            )
        lines.append("最近连续消息：")
        if truncated_count:
            lines.append(f"注意：由于消息数量超过上限，前面有 {truncated_count} 条较早消息未包含。")
        for idx, item in enumerate(messages, start=1):
            attachments = f" 附件/媒体: {', '.join(item.attachments)}" if item.attachments else ""
            text = item.text or item.outline or "(无文本)"
            role = "管理员" if item.is_admin else "群友"
            lines.append(f"{idx}. {item.sender_name or item.sender_id}({role}): {text}{attachments}")
        recent_answer = self._recent_answer_for(event, project)
        if recent_answer:
            lines.extend(
                [
                    "最近一次机器人回答，供判断管理员纠错上下文：",
                    recent_answer.answer,
                ]
            )
        lines.append(
            "请先判断是否需要回复。如果问题已被其他群友解答或已解决，返回 reply=false。"
            "如果需要回复，优先查 QA Markdown；QA 不足时再调查仓库。"
            "涉及功能行为、页面操作、状态流转或报错原因时，回答前要尽量确认项目里的真实实现，不要只给通用经验。"
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
            "reason": str(data.get("reason") or "").strip(),
        }

    async def _provider_id_for_event(self, event: AstrMessageEvent) -> Any:
        if hasattr(self.context, "get_current_chat_provider_id"):
            try:
                provider_id = await self.context.get_current_chat_provider_id(event.unified_msg_origin)
            except Exception:
                logger.debug("Project Helper failed to resolve current chat provider", exc_info=True)
            else:
                if provider_id:
                    return provider_id
        provider = self.context.get_using_provider(None)
        meta = provider.meta() if provider is not None and hasattr(provider, "meta") else None
        provider_id = getattr(meta, "id", None)
        if not provider_id:
            raise RuntimeError("当前会话没有可用的 LLM provider。")
        return provider_id

    async def _should_suppress_after_followup(
        self,
        event: AstrMessageEvent,
        project: ProjectBinding,
        original_messages: list[BufferedMessage],
        answer: str,
        followup_messages: list[BufferedMessage],
        trace: AgentTraceRecorder | None,
    ) -> bool:
        relevant_followups = [
            item
            for item in followup_messages
            if item.sender_id != event.get_self_id() and (item.text or item.outline or item.attachments)
        ]
        if not relevant_followups:
            if trace:
                trace.post_check = {"decision": "no_followup", "reason": "Agent 运行期间没有新的群消息。"}
            return False

        fallback_decision, fallback_reason = self._heuristic_followup_decision(relevant_followups)
        if fallback_decision == "suppress":
            if trace:
                trace.post_check = {
                    "decision": "suppress",
                    "reason": fallback_reason,
                    "method": "heuristic",
                }
            return True

        try:
            provider_id = await self._provider_id_for_event(event)
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    system_prompt=(
                        "你只判断一个项目群答疑机器人是否还应该发送已经准备好的回复。"
                        "如果新消息已经把原问题解决、提问者表示解决/不用了、管理员要求别回，输出 suppress。"
                        "如果新消息只是补充日志、追问、催促、无关闲聊，输出 keep。"
                        "只能输出 JSON：{\"decision\":\"keep|suppress\",\"reason\":\"简短原因\"}"
                    ),
                    prompt=self._build_followup_check_prompt(
                        project,
                        original_messages,
                        answer,
                        relevant_followups,
                    ),
                ),
                timeout=min(30.0, self.config.agent_timeout_seconds),
            )
            raw = str(getattr(response, "completion_text", "") or "").strip()
            data = self._parse_followup_json(raw)
            decision = str(data.get("decision") or "keep")
            reason = str(data.get("reason") or "")
            if trace:
                trace.post_check = {
                    "decision": decision,
                    "reason": reason,
                    "method": "llm",
                    "raw": _clip_text(raw, 1200),
                }
            return decision == "suppress"
        except Exception as exc:
            logger.warning("Project Helper followup suppression check failed: %s", exc)
            if trace:
                trace.post_check = {
                    "decision": "keep",
                    "reason": f"followup check failed: {exc}; fallback={fallback_decision}",
                    "method": "fallback",
                }
            return fallback_decision == "suppress"

    def _build_followup_check_prompt(
        self,
        project: ProjectBinding,
        original_messages: list[BufferedMessage],
        answer: str,
        followup_messages: list[BufferedMessage],
    ) -> str:
        lines = [
            f"项目：{project.label()}",
            "原始问题消息：",
        ]
        for idx, item in enumerate(original_messages, start=1):
            lines.append(f"{idx}. {item.sender_name or item.sender_id}: {item.text or item.outline}")
        lines.extend(
            [
                "机器人准备发送的回答：",
                answer,
                "Agent 调查期间群里的新消息：",
            ]
        )
        for idx, item in enumerate(followup_messages, start=1):
            role = "管理员" if item.is_admin else "群友"
            attachments = f" 附件/媒体: {', '.join(item.attachments)}" if item.attachments else ""
            lines.append(f"{idx}. {item.sender_name or item.sender_id}({role}): {item.text or item.outline}{attachments}")
        lines.append("判断机器人是否还应该发送这条准备好的回答。")
        return "\n".join(lines)

    def _parse_followup_json(self, raw: str) -> dict[str, str]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = ANSWER_JSON_RE.search(raw)
            if not match:
                return {"decision": "keep", "reason": "followup checker returned non-JSON"}
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"decision": "keep", "reason": "followup checker returned invalid JSON"}
        if not isinstance(data, dict):
            return {"decision": "keep", "reason": "followup checker returned non-object JSON"}
        decision = str(data.get("decision") or "keep").strip().lower()
        if decision not in {"keep", "suppress"}:
            decision = "keep"
        return {"decision": decision, "reason": str(data.get("reason") or "").strip()}

    def _heuristic_followup_decision(self, followup_messages: list[BufferedMessage]) -> tuple[str, str]:
        text = "\n".join((item.text or item.outline or "") for item in followup_messages).lower()
        if not text:
            return "keep", "只有附件或空文本，保留回复。"
        solved_patterns = [
            "解决了",
            "已解决",
            "好了",
            "可以了",
            "不用了",
            "不用回",
            "别回",
            "不用管",
            "懂了",
            "明白了",
            "确实",
            "是这个原因",
        ]
        if any(pattern in text for pattern in solved_patterns):
            return "suppress", "新消息疑似表示问题已解决或无需回复。"
        return "keep", "未发现明确已解决信号。"

    def _should_replay_followups(self, followup_messages: list[BufferedMessage]) -> bool:
        if not followup_messages:
            return False
        text = "\n".join(item.text or item.outline or "" for item in followup_messages).strip()
        if any(item.attachments for item in followup_messages):
            return True
        if not text:
            return False
        question_markers = ("?", "？", "怎么", "为啥", "为什么", "咋", "能不能", "可以吗", "怎么办", "报错", "失败", "不行", "没有", "获取不到")
        supplement_markers = ("日志", "截图", "报错", "错误", "traceback", "error", "warning", "补充", "还有", "另外", "刚才", "复现")
        return any(marker in text for marker in (*question_markers, *supplement_markers))

    def _project_for_event(self, event: AstrMessageEvent) -> ProjectBinding | None:
        return self._project_for_group(event.get_group_id())

    def _project_for_group(self, group_id: str) -> ProjectBinding | None:
        group_id = str(group_id or "").strip()
        if not group_id:
            return None
        for project in self.config.projects:
            if not project.enabled:
                continue
            if project.group_id == group_id:
                return project
        return None

    def _project_key(self, project: ProjectBinding) -> str:
        return project.group_id or project.repo_path or project.repo_url or project.label()

    def _safe_project_trace_name(self, project: ProjectBinding) -> str:
        raw = self._project_key(project)
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
        return sanitized or "project"

    def _latest_trace_path(self, project: ProjectBinding) -> Path:
        return self.trace_base_dir / self._safe_project_trace_name(project) / "latest.json"

    def _trace_archive_path(self, project: ProjectBinding, run_id: str) -> Path:
        return self.trace_base_dir / self._safe_project_trace_name(project) / f"{run_id}.json"

    def _lock_for_project(self, project: ProjectBinding) -> asyncio.Lock:
        key = self._project_key(project)
        lock = self._repo_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._repo_locks[key] = lock
        return lock

    def _should_watch(self, event: AstrMessageEvent, project: ProjectBinding | None) -> bool:
        if project is None:
            return False
        if not project.repo_url:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.get_sender_id() == event.get_self_id():
            return False
        return True

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        try:
            return bool(event.is_admin())
        except Exception:
            return False

    def _append_buffer_message(
        self,
        target: list[BufferedMessage],
        message: BufferedMessage,
        *,
        running: bool,
        buf: SessionBuffer,
    ) -> None:
        target.append(message)
        if len(target) <= self.config.max_buffer_messages:
            return
        overflow = len(target) - self.config.max_buffer_messages
        del target[:overflow]
        if running:
            buf.running_truncated_count += overflow
        else:
            buf.truncated_count += overflow

    def _drop_idle_buffer(
        self,
        event: AstrMessageEvent,
        session_id: str,
        buf: SessionBuffer,
        *,
        replay_messages: list[BufferedMessage] | None = None,
        replay_truncated_count: int = 0,
    ) -> None:
        if buf.processing:
            return
        if replay_messages:
            buf.messages = [*buf.messages, *replay_messages]
            if len(buf.messages) > self.config.max_buffer_messages:
                overflow = len(buf.messages) - self.config.max_buffer_messages
                del buf.messages[:overflow]
                buf.truncated_count += overflow
            buf.truncated_count += replay_truncated_count
            buf.updated_at = time.time()
        if buf.messages:
            if not buf.task or buf.task.done():
                buf.task = asyncio.create_task(self._delayed_process(event, session_id))
            return
        if not buf.running_messages:
            self.buffers.pop(session_id, None)

    def _write_trace(
        self,
        event: AstrMessageEvent,
        project: ProjectBinding,
        repo_root: Path,
        trace: AgentTraceRecorder,
        trigger_messages: list[BufferedMessage],
        trigger_truncated_count: int,
        followup_messages: list[BufferedMessage],
        followup_truncated_count: int,
    ) -> None:
        payload = trace.to_payload(
            event=event,
            project=project,
            repo_root=repo_root,
            qa_path=self._qa_path(project),
            trigger_messages=trigger_messages,
            trigger_truncated_count=trigger_truncated_count,
            followup_messages=followup_messages,
            followup_truncated_count=followup_truncated_count,
        )
        latest_path = self._latest_trace_path(project)
        archive_path = self._trace_archive_path(project, trace.run_id)
        try:
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(payload, ensure_ascii=False, indent=2)
            latest_path.write_text(data, encoding="utf-8")
            archive_path.write_text(data, encoding="utf-8")
        except Exception:
            logger.warning("Project Helper failed to write trace", exc_info=True)

    async def _build_buffered_message(self, event: AstrMessageEvent) -> BufferedMessage:
        attachments = []
        for comp in event.get_messages():
            desc = await self._describe_component(comp)
            if desc:
                attachments.append(desc)
        return BufferedMessage(
            sender_id=event.get_sender_id(),
            sender_name=event.get_sender_name(),
            is_admin=self._is_admin(event),
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

    def _repo_root(self, project: ProjectBinding) -> Path:
        configured = Path(project.repo_path or self._default_repo_path(project)).expanduser()
        if configured.is_absolute():
            return configured.resolve()
        return (self.repo_base_dir / configured).resolve()

    def _qa_path(self, project: ProjectBinding) -> Path:
        configured = Path(project.qa_path or f"{self._default_repo_path(project)}_QA.md").expanduser()
        if configured.is_absolute():
            return configured.resolve()
        return (self.data_dir / "qa" / configured).resolve()

    def _default_repo_path(self, project: ProjectBinding) -> str:
        name = _repo_name_from_url(project.repo_url) or project.group_id or "target_repo"
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
        return sanitized or "target_repo"

    async def _ensure_repo(self, project: ProjectBinding, *, update: bool) -> Path:
        async with self._lock_for_project(project):
            repo_root = self._repo_root(project)
            if repo_root.exists() and (repo_root / ".git").exists():
                if update:
                    await asyncio.to_thread(self._git_update, project, repo_root)
                return repo_root

            if repo_root.exists() and not (repo_root / ".git").exists():
                if project.repo_url:
                    raise RuntimeError(f"repo_path exists but is not a git repository: {repo_root}")
                return repo_root

            if not project.repo_url:
                raise RuntimeError("repo_url 未配置，且 repo_path 不存在。请先配置目标仓库或运行本地 checkout。")

            repo_root.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._git_clone, project, repo_root)
            return repo_root

    def _git_clone(self, project: ProjectBinding, repo_root: Path) -> None:
        if shutil.which("git") is None:
            raise RuntimeError("系统找不到 git，无法克隆仓库。")
        cmd = ["git", "clone", "--depth", "1"]
        branch = project.repo_branch or self._detect_preferred_branch(project.repo_url)
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([project.repo_url, str(repo_root)])
        self._run_cmd(cmd, cwd=self.repo_base_dir)

    def _git_update(self, project: ProjectBinding, repo_root: Path) -> None:
        if shutil.which("git") is None:
            raise RuntimeError("系统找不到 git，无法更新仓库。")
        self._run_cmd(["git", "fetch", "--all", "--prune"], cwd=repo_root)
        if project.repo_branch:
            self._run_cmd(["git", "checkout", project.repo_branch], cwd=repo_root)
        self._run_cmd(["git", "pull", "--ff-only"], cwd=repo_root)

    def _detect_preferred_branch(self, repo_url: str) -> str:
        if shutil.which("git") is None:
            raise RuntimeError("系统找不到 git，无法检查仓库分支。")
        for branch in ("main", "master"):
            if self._remote_branch_exists(repo_url, branch):
                return branch
        return ""

    def _remote_branch_exists(self, repo_url: str, branch: str) -> bool:
        env = os.environ.copy()
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        proc = subprocess.run(
            ["git", "ls-remote", "--exit-code", "--heads", repo_url, branch],
            cwd=str(self.repo_base_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode == 0:
            return True
        if proc.returncode == 2:
            return False
        stderr = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(stderr[:1000] or f"无法检查远端分支 {branch}")

    async def _notify_admins(self, event: AstrMessageEvent, text: str) -> None:
        if not self.config.error_notify_admins:
            return
        if not self.config.admin_qqs:
            logger.warning("Project Helper failed but admin_qqs is empty: %s", text)
            return

        platform_id = event.get_platform_id()
        if not platform_id:
            logger.warning("Project Helper cannot notify admins without platform id: %s", text)
            return

        message = MessageChain([Plain(text)])
        for admin_qq in self.config.admin_qqs:
            session = f"{platform_id}:{MessageType.FRIEND_MESSAGE.value}:{admin_qq}"
            try:
                sent = await self.context.send_message(session, message)
                if not sent:
                    logger.warning("Project Helper admin notification was not sent to %s", session)
            except Exception as exc:
                logger.warning("Project Helper admin notification failed for %s: %s", session, exc)

    def _remember_answer(
        self,
        event: AstrMessageEvent,
        project: ProjectBinding,
        messages: list[BufferedMessage],
        answer: str,
    ) -> None:
        self._prune_recent_answers()
        self.recent_answers[event.unified_msg_origin] = RecentAnswer(
            project_key=self._project_key(project),
            answer=answer,
            question_context="\n".join(item.text or item.outline for item in messages if item.text or item.outline),
            created_at=time.time(),
        )

    def _recent_answer_for(self, event: AstrMessageEvent, project: ProjectBinding) -> RecentAnswer | None:
        self._prune_recent_answers()
        answer = self.recent_answers.get(event.unified_msg_origin)
        if answer and answer.project_key == self._project_key(project):
            return answer
        return None

    def _prune_recent_answers(self) -> None:
        now = time.time()
        expired = [
            session_id
            for session_id, answer in self.recent_answers.items()
            if now - answer.created_at > RECENT_ANSWER_TTL_SECONDS
        ]
        for session_id in expired:
            self.recent_answers.pop(session_id, None)

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
