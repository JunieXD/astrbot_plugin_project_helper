# AstrBot Project Helper

让 AstrBot 在项目交流群里做“会读代码的答疑群友”。

插件会监听群内连续消息，先交给 Agent 判断是否和目标 GitHub 项目有关；如果只是闲聊，它返回 `reply=false`，机器人不发言；如果是项目问题，Agent 会用只读仓库工具检索代码和 Markdown，然后把答案发回群里。

## 为什么先用 AstrBot 内置 Agent

当前版本优先使用 AstrBot 自带的 `Context.tool_loop_agent()`，并给它注入 4 个只读仓库工具：

- `repo_tree`: 查看目录结构
- `repo_search`: 搜索代码/文档
- `repo_read_file`: 按行读取文件
- `repo_find_files`: 按 glob 找文件

这已经满足“能自主读取项目代码以及 Markdown 文件并回答”的核心要求，而且不用额外接入外部 Agent 服务。

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

核心配置在 WebUI 里填：

- `repo_url`: 目标项目 Git 地址，例如 `https://github.com/owner/repo.git`
- `repo_branch`: 可选分支/tag/commit
- `repo_path`: 本地 checkout 路径。相对路径会放到 `data/plugin_data/astrbot_plugin_project_helper/repos/`
- `enabled_sessions`: 限定处理哪些群。空列表表示所有会话
- `buffer_seconds`: 多条消息聚合等待时间
- `auto_update_repo`: 每次回答前是否自动 `git fetch/pull`

常用命令：

```text
/project_helper_status
/project_helper_update
```

`/project_helper_update` 需要管理员权限。

## 当前限制

- 图片和日志文件目前会作为“附件/媒体提示”进入上下文，还没有自动 OCR 或完整文件内容抽取。
- Agent 最终必须输出 JSON；如果模型不遵守，插件会把原始文本当作回答兜底发送。
- 仓库工具是只读的，不暴露 shell/write/edit，避免群消息触发任意本地操作。
