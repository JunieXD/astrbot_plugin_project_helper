# AstrBot Project Helper

让 AstrBot 在项目交流群里做“会读代码的答疑群友”。

插件会监听群内连续消息，先交给 Agent 判断是否和目标 GitHub 项目有关；如果只是闲聊、问题已经被其他群友解答，或提问者表示已解决，它返回 `reply=false`，机器人不发言；如果是项目问题，Agent 会先查项目 QA Markdown，再按需用只读仓库工具检索代码和 Markdown，然后把答案发回群里。

## 为什么先用 AstrBot 内置 Agent

当前版本优先使用 AstrBot 自带的 `Context.tool_loop_agent()`，并给它注入只读仓库工具和一个受控 QA Markdown 记忆工具：

- `repo_tree`: 查看目录结构
- `repo_search`: 搜索代码/文档
- `repo_read_file`: 按行读取文件
- `repo_find_files`: 按 glob 找文件
- `qa_read`: 读取项目常见 QA
- `qa_search`: 搜索项目常见 QA
- `qa_upsert`: 追加一条调查后的 QA 记录

这已经满足“能自主读取项目代码以及 Markdown 文件并回答”的核心要求，而且不用额外接入外部 Agent 服务。QA 工具只允许读写当前项目绑定的单个 Markdown 文件，不暴露任意写文件能力。

PI Agent 是更完整的本地 coding agent，但默认权限边界更宽，接入 AstrBot 还需要额外进程管理、会话协议和权限隔离。等内置 Agent 的效果确实不够时，再把 PI Agent 做成可选 runner 更合适。

## 参考 MaiBot 的地方

MaiBot 的关键不是“每条消息都回”，而是先聚合最近消息，再做回复必要性判断，必要时才进入后续规划和回复。本插件采用同样思路：

- 用 `buffer_seconds` 等待群友连续发完多条消息
- 聚合文本、图片/文件/引用等占位信息
- 由 Agent 统一判断是否值得回复
- 不相关时完全沉默，避免刷屏

## 安装

把本仓库放到 AstrBot 的插件目录，例如：

```bash
cd /path/to/AstrBot/data/plugins
git clone https://github.com/JunieXD/astrbot_plugin_project_helper.git
```

然后在 AstrBot WebUI 启用插件。插件没有第三方 Python 依赖，只需要系统有 `git`。

仓库名、`metadata.yaml` 的 `name` 字段、插件目录名和 Python 包名都保持为
`astrbot_plugin_project_helper`，AstrBot 从 GitHub 安装后会按这个顶层包加载：
`data.plugins.astrbot_plugin_project_helper.main`。

## 配置

核心配置在 WebUI 里填。`projects` 是项目绑定列表，一条配置表示一个 QQ 群绑定一个 GitHub 项目：

- `group_id`: QQ 群号，只填数字，例如 `123456789`
- `project_prompt`: 项目简介提示词，用来告诉 Agent 这个项目做什么、常见场景、哪些问题算本项目问题
- `repo_url`: 目标项目 Git 地址，必填，例如 `https://github.com/owner/repo.git`
- `repo_branch`: 可选分支/tag/commit；留空时克隆会优先尝试 `main`，其次 `master`，再退回远端默认分支
- `repo_path`: 本地 checkout 路径。相对路径会放到 `data/plugin_data/astrbot_plugin_project_helper/repos/`
- `qa_path`: 项目 QA Markdown 路径。相对路径会放到 `data/plugin_data/astrbot_plugin_project_helper/qa/`
- `admin_qqs`: 处理失败时私聊通知的管理员 QQ 号列表；未配置则只写日志，不会在群里报错
- `buffer_seconds`: 多条消息聚合等待时间，默认 15 秒
- `max_buffer_messages`: 一次聚合最多保留的消息数，默认 20 条
- `max_answer_chars`: 单次群回复最大字符数，默认 700，避免群聊里输出长篇
- `answer_style_prompt`: 群聊回复语气提示词，默认会要求 Agent 像熟悉项目的真人群友一样短句回复
- `max_tool_calls`: 单次 Agent 最多工具调用轮数，默认 25
- `send_typing`: Agent 调查期间显示平台的“正在输入”状态，不会额外发送文字消息
- `auto_update_repo`: 每次回答前是否自动 `git fetch/pull`，默认开启
- `include_sources`: 是否要求回答末尾附简短文件依据，默认关闭

项目显示名不需要单独配置，插件会直接使用 GitHub 仓库名。首次处理对应群的问题或执行 `/ph update` 时，插件会把仓库克隆到本地 `repo_path`；后续 Agent 通过只读工具检索这个本地 checkout 的代码和 Markdown。

QA Markdown 会在第一次产生有效群回复时自动创建并记录一条问答。Agent 也可以在调查后主动用 `qa_upsert` 写入更完整的结论；如果它忘了写，插件会把已发送的群回复作为兜底记录保存下来。

管理员命令：

```text
/ph status
/ph update
```

两个命令都需要 AstrBot 管理员权限。`/ph status` 会显示当前群绑定、仓库路径、QA 路径和关键运行参数；`/ph update` 会立即 clone 或更新当前群绑定的仓库。

## 当前限制

- 图片和日志文件目前会作为“附件/媒体提示”进入上下文，还没有自动 OCR 或完整文件内容抽取。
- Agent 最终必须输出 JSON；如果模型不遵守，插件会把原始文本当作回答兜底发送。
- 仓库工具是只读的，不暴露 shell/write/edit；QA 工具只允许写当前项目的 QA Markdown。
