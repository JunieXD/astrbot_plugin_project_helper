from __future__ import annotations

import asyncio
import fnmatch
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from astrbot.core.agent.tool import FunctionTool


DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
}

DEFAULT_TEXT_EXTENSIONS = {
    ".bat",
    ".c",
    ".cc",
    ".cfg",
    ".cmd",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".csv",
    ".dockerfile",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".lua",
    ".m",
    ".md",
    ".mdx",
    ".php",
    ".plist",
    ".proto",
    ".py",
    ".rb",
    ".rs",
    ".rst",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class RepoToolLimits:
    max_files: int = 200
    max_search_matches: int = 80
    max_read_bytes: int = 48_000
    max_tree_depth: int = 4
    max_output_chars: int = 20_000


@dataclass(frozen=True)
class QAMemoryLimits:
    max_read_bytes: int = 64_000
    max_search_matches: int = 20
    max_entry_chars: int = 12_000
    max_output_chars: int = 20_000


class QAMemoryTools:
    def __init__(
        self,
        qa_path: Path,
        *,
        project_label: str,
        limits: QAMemoryLimits | None = None,
    ) -> None:
        self.qa_path = qa_path.resolve()
        self.project_label = project_label
        self.limits = limits or QAMemoryLimits()

    def tool_set(self) -> list[FunctionTool]:
        return [
            FunctionTool(
                name="qa_read",
                description=(
                    "Read the project QA Markdown memory. Use this before searching code "
                    "to see whether the question has already been investigated."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                },
                handler=self.qa_read,
            ),
            FunctionTool(
                name="qa_search",
                description=(
                    "Search the QA Markdown memory for existing answers. "
                    "Use concise keywords from the user's question or error message."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keyword or phrase to search in QA memory.",
                        }
                    },
                    "required": ["query"],
                },
                handler=self.qa_search,
            ),
            FunctionTool(
                name="qa_upsert",
                description=(
                    "Append a corrected or reusable QA entry to the project QA Markdown memory. "
                    "Use this after investigating an answer that is likely to help future users, "
                    "or after correcting a previous wrong answer. Keep entries concise."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Short natural-language question or issue title.",
                        },
                        "answer": {
                            "type": "string",
                            "description": "Reusable answer for ordinary project users.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "Optional short evidence, such as docs/code paths and line numbers.",
                        },
                        "tags": {
                            "type": "string",
                            "description": "Optional comma-separated keywords.",
                        },
                    },
                    "required": ["question", "answer"],
                },
                handler=self.qa_upsert,
            ),
        ]

    async def qa_read(self, **_: object) -> str:
        if not self.qa_path.exists():
            return self._clip(self._initial_content())
        data = await asyncio.to_thread(self._read_text)
        return self._clip(data)

    async def qa_search(self, query: str, **_: object) -> str:
        query = str(query or "").strip()
        if not query:
            return "Search query is empty."
        if not self.qa_path.exists():
            return "QA memory file does not exist yet."

        lowered_terms = [term for term in query.lower().split() if term]
        if not lowered_terms:
            lowered_terms = [query.lower()]

        text = await asyncio.to_thread(self._read_text)
        lines = text.splitlines()
        matches: list[str] = []
        for idx, line in enumerate(lines, start=1):
            lowered = line.lower()
            if any(term in lowered for term in lowered_terms):
                start = max(1, idx - 2)
                end = min(len(lines), idx + 4)
                snippet = "\n".join(
                    f"{line_no}: {lines[line_no - 1]}"
                    for line_no in range(start, end + 1)
                )
                matches.append(snippet)
                if len(matches) >= self.limits.max_search_matches:
                    break
        if not matches:
            return "No QA memory matches."
        if len(matches) >= self.limits.max_search_matches:
            matches.append(f"... truncated after {self.limits.max_search_matches} matches")
        return self._clip("\n\n---\n\n".join(matches))

    async def qa_upsert(
        self,
        question: str,
        answer: str,
        evidence: str = "",
        tags: str = "",
        **_: object,
    ) -> str:
        question = str(question or "").strip()
        answer = str(answer or "").strip()
        evidence = str(evidence or "").strip()
        tags = str(tags or "").strip()
        if not question:
            return "question is required."
        if not answer:
            return "answer is required."

        entry = self._format_entry(question, answer, evidence, tags)
        if len(entry) > self.limits.max_entry_chars:
            return f"QA entry is too long; keep it under {self.limits.max_entry_chars} characters."

        await asyncio.to_thread(self._append_entry, entry)
        return f"QA memory updated: {self.qa_path.name}"

    def _initial_content(self) -> str:
        return (
            f"# {self.project_label} 常见 QA\n\n"
            "这个文件由 AstrBot 项目答疑助手维护，用来沉淀已经调查过的问题。\n\n"
            "## 条目\n"
        )

    def _format_entry(self, question: str, answer: str, evidence: str, tags: str) -> str:
        date_text = time.strftime("%Y-%m-%d")
        parts = [
            "",
            f"### {question}",
            "",
            f"- 更新时间：{date_text}",
        ]
        if tags:
            parts.append(f"- 标签：{tags}")
        parts.extend(
            [
                "",
                "#### 答案",
                "",
                answer,
            ]
        )
        if evidence:
            parts.extend(
                [
                    "",
                    "#### 依据",
                    "",
                    evidence,
                ]
            )
        parts.append("")
        return "\n".join(parts)

    def _read_text(self) -> str:
        with self.qa_path.open("r", encoding="utf-8", errors="replace") as handle:
            data = handle.read(self.limits.max_read_bytes + 1)
        if len(data.encode("utf-8", errors="replace")) > self.limits.max_read_bytes:
            return data[: self.limits.max_read_bytes] + "\n... truncated; search for narrower keywords"
        return data

    def _append_entry(self, entry: str) -> None:
        self.qa_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.qa_path.exists():
            self.qa_path.write_text(self._initial_content(), encoding="utf-8")
        with self.qa_path.open("a", encoding="utf-8") as handle:
            handle.write(entry)

    def _clip(self, text: str) -> str:
        if len(text) <= self.limits.max_output_chars:
            return text
        return text[: self.limits.max_output_chars] + f"\n... truncated at {self.limits.max_output_chars} characters"


class RepositoryTools:
    def __init__(
        self,
        repo_root: Path,
        *,
        limits: RepoToolLimits | None = None,
        ignore_dirs: Iterable[str] | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.limits = limits or RepoToolLimits()
        self.ignore_dirs = set(ignore_dirs or DEFAULT_IGNORE_DIRS)

    def tool_set(self) -> list[FunctionTool]:
        return [
            FunctionTool(
                name="repo_tree",
                description=(
                    "List files and directories in the target repository. Use this first "
                    "to understand project structure. Paths are relative to repository root."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path. Empty means repository root.",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Maximum recursion depth from the path.",
                        },
                    },
                },
                handler=self.repo_tree,
            ),
            FunctionTool(
                name="repo_search",
                description=(
                    "Search text in repository files using a literal or regular expression query. "
                    "Returns file paths, line numbers, and short matching lines."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional relative file or directory to search in.",
                        },
                        "regex": {
                            "type": "boolean",
                            "description": "Whether query should be treated as a regular expression.",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether matching is case-sensitive.",
                        },
                    },
                    "required": ["query"],
                },
                handler=self.repo_search,
            ),
            FunctionTool(
                name="repo_read_file",
                description=(
                    "Read a text file from the repository. Use line ranges for large files. "
                    "The tool refuses paths outside the repository."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path.",
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "1-based start line. Defaults to 1.",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "1-based inclusive end line.",
                        },
                    },
                    "required": ["path"],
                },
                handler=self.repo_read_file,
            ),
            FunctionTool(
                name="repo_find_files",
                description=(
                    "Find files by glob pattern, such as '*.py', 'docs/**/*.md', or '*config*'."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern matched against relative file paths and file names.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional relative directory to search in.",
                        },
                    },
                    "required": ["pattern"],
                },
                handler=self.repo_find_files,
            ),
        ]

    async def repo_tree(
        self,
        path: str = "",
        depth: int | None = None,
        **_: object,
    ) -> str:
        root = self._safe_path(path or "")
        if not root.exists():
            return f"Path not found: {path}"
        if not root.is_dir():
            return f"Not a directory: {path}"

        max_depth = self._bounded_int(depth, self.limits.max_tree_depth, 0, 12)
        lines: list[str] = []
        count = 0

        def walk(current: Path, current_depth: int) -> None:
            nonlocal count
            if count >= self.limits.max_files:
                return
            try:
                children = sorted(
                    current.iterdir(),
                    key=lambda item: (not item.is_dir(), item.name.lower()),
                )
            except OSError as exc:
                lines.append(f"{self._rel(current)}/  [error: {exc}]")
                return

            for child in children:
                if count >= self.limits.max_files:
                    break
                if self._ignored(child):
                    continue
                rel = self._rel(child)
                suffix = "/" if child.is_dir() else ""
                lines.append(f"{rel}{suffix}")
                count += 1
                if child.is_dir() and current_depth < max_depth:
                    walk(child, current_depth + 1)

        walk(root, 0)
        if count >= self.limits.max_files:
            lines.append(f"... truncated after {self.limits.max_files} entries")
        return self._clip("\n".join(lines) if lines else "(empty)")

    async def repo_find_files(
        self,
        pattern: str,
        path: str = "",
        **_: object,
    ) -> str:
        base = self._safe_path(path or "")
        if not base.exists():
            return f"Path not found: {path}"
        patterns = [pattern, f"**/{pattern}" if "/" not in pattern else pattern]
        matches: list[str] = []

        for file_path in self._iter_files(base):
            rel = self._rel(file_path)
            name = file_path.name
            if any(fnmatch.fnmatch(rel, item) or fnmatch.fnmatch(name, item) for item in patterns):
                matches.append(rel)
                if len(matches) >= self.limits.max_files:
                    break

        if len(matches) >= self.limits.max_files:
            matches.append(f"... truncated after {self.limits.max_files} files")
        return self._clip("\n".join(matches) if matches else "No files matched.")

    async def repo_search(
        self,
        query: str,
        path: str = "",
        regex: bool = False,
        case_sensitive: bool = False,
        **_: object,
    ) -> str:
        if not query:
            return "Search query is empty."
        base = self._safe_path(path or "")
        if not base.exists():
            return f"Path not found: {path}"

        rg = self._which_rg()
        if rg:
            return await self._repo_search_rg(
                rg,
                query=query,
                base=base,
                regex=regex,
                case_sensitive=case_sensitive,
            )
        return await asyncio.to_thread(
            self._repo_search_python,
            query,
            base,
            regex,
            case_sensitive,
        )

    async def repo_read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **_: object,
    ) -> str:
        file_path = self._safe_path(path)
        if not file_path.exists():
            return f"File not found: {path}"
        if not file_path.is_file():
            return f"Not a file: {path}"
        if not self._looks_text(file_path):
            return f"Refusing to read likely binary file: {path}"

        start = self._bounded_int(start_line, 1, 1, 10_000_000)
        end = self._bounded_int(end_line, 0, 0, 10_000_000)
        if end and end < start:
            return "end_line must be greater than or equal to start_line."

        data = await asyncio.to_thread(self._read_text_slice, file_path, start, end)
        return self._clip(data)

    def _read_text_slice(self, file_path: Path, start: int, end: int) -> str:
        lines: list[str] = []
        total_bytes = 0
        truncated = False
        with file_path.open("r", encoding="utf-8", errors="replace") as handle:
            for idx, line in enumerate(handle, start=1):
                if idx < start:
                    continue
                if end and idx > end:
                    break
                item = f"{idx}: {line.rstrip()}"
                total_bytes += len(item.encode("utf-8", errors="replace"))
                if total_bytes > self.limits.max_read_bytes:
                    truncated = True
                    break
                lines.append(item)
        if truncated:
            lines.append(f"... truncated at {self.limits.max_read_bytes} bytes; use a narrower line range")
        return "\n".join(lines) if lines else "(no lines in requested range)"

    async def _repo_search_rg(
        self,
        rg: str,
        *,
        query: str,
        base: Path,
        regex: bool,
        case_sensitive: bool,
    ) -> str:
        cmd = [
            rg,
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(self.limits.max_search_matches),
        ]
        if not regex:
            cmd.append("--fixed-strings")
        if not case_sensitive:
            cmd.append("--ignore-case")
        for ignored in sorted(self.ignore_dirs):
            cmd.extend(["--glob", f"!{ignored}/**"])
        cmd.extend([query, str(base)])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.repo_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
        except asyncio.TimeoutError:
            return "Search timed out; try a narrower query or path."

        if proc.returncode not in (0, 1):
            return f"Search failed: {stderr.decode('utf-8', errors='replace')[:1000]}"

        text = stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return "No matches."

        lines = []
        for line in text.splitlines()[: self.limits.max_search_matches]:
            parts = line.split(":", 2)
            if len(parts) == 3:
                file_part = self._display_rg_path(parts[0])
                lines.append(f"{file_part}:{parts[1]}: {parts[2].strip()}")
            else:
                lines.append(line)
        if len(text.splitlines()) >= self.limits.max_search_matches:
            lines.append(f"... truncated after {self.limits.max_search_matches} matches")
        return self._clip("\n".join(lines))

    def _repo_search_python(
        self,
        query: str,
        base: Path,
        regex: bool,
        case_sensitive: bool,
    ) -> str:
        import re

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(query if regex else re.escape(query), flags)
        results: list[str] = []

        for file_path in self._iter_files(base):
            if not self._looks_text(file_path):
                continue
            try:
                with file_path.open("r", encoding="utf-8", errors="replace") as handle:
                    for idx, line in enumerate(handle, start=1):
                        if compiled.search(line):
                            results.append(f"{self._rel(file_path)}:{idx}: {line.strip()}")
                            if len(results) >= self.limits.max_search_matches:
                                results.append(
                                    f"... truncated after {self.limits.max_search_matches} matches"
                                )
                                return self._clip("\n".join(results))
            except OSError:
                continue
        return self._clip("\n".join(results) if results else "No matches.")

    def _iter_files(self, base: Path) -> Iterable[Path]:
        if base.is_file():
            if not self._ignored(base):
                yield base
            return
        for root, dirs, files in os.walk(base):
            root_path = Path(root)
            dirs[:] = [item for item in dirs if item not in self.ignore_dirs]
            if self._ignored(root_path):
                continue
            for filename in sorted(files):
                file_path = root_path / filename
                if not self._ignored(file_path):
                    yield file_path

    def _safe_path(self, relative_path: str) -> Path:
        raw = Path(str(relative_path or "."))
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (self.repo_root / raw).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"Path escapes repository root: {relative_path}") from exc
        return candidate

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.repo_root).as_posix()

    def _ignored(self, path: Path) -> bool:
        try:
            relative_parts = path.resolve().relative_to(self.repo_root).parts
        except ValueError:
            relative_parts = path.parts
        return any(part in self.ignore_dirs for part in relative_parts)

    def _looks_text(self, path: Path) -> bool:
        if path.suffix.lower() in DEFAULT_TEXT_EXTENSIONS:
            return True
        try:
            with path.open("rb") as handle:
                chunk = handle.read(2048)
        except OSError:
            return False
        return b"\0" not in chunk

    def _display_rg_path(self, path: str) -> str:
        item = Path(path)
        if item.is_absolute():
            try:
                return item.resolve().relative_to(self.repo_root).as_posix()
            except ValueError:
                return path
        return item.as_posix()

    def _which_rg(self) -> str | None:
        try:
            proc = subprocess.run(
                ["which", "rg"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return None
        result = proc.stdout.strip()
        return result or None

    def _bounded_int(self, value: int | None, default: int, minimum: int, maximum: int) -> int:
        if value is None:
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))

    def _clip(self, text: str) -> str:
        if len(text) <= self.limits.max_output_chars:
            return text
        return (
            text[: self.limits.max_output_chars]
            + f"\n... truncated at {self.limits.max_output_chars} characters"
        )
