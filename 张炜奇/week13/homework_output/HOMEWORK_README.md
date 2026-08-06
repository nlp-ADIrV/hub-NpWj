## 作业目标

写一套可以实现渐进式加载执行skills的harness。


## 新增文件 - harness模块

### src/skill_loader.py — 渐进式技能加载器

这是整个 harness 的核心，负责管理 skill 的完整生命周期。

**Skill 发现与元信息解析**：启动时扫描 `skills/` 目录，找到所有包含 `SKILL.md` 的子目录。仅解析每个文件的 YAML frontmatter（`---` 之间的部分），提取 name 和 description 两个字段。这个阶段不加载完整文件正文。

**触发关键词提取**：从 description 字段中自动提取触发关键词。支持中文引号短语、英文引号短语、中文关键词、英文长词等多种模式，并过滤掉常用停用词。例如 flash-card 的 description 里包含"给我做张 crazy 词的闪卡"，这个短语就被自动提取为触发词。

**关键词匹配**：用户输入后，遍历所有 skill 的触发关键词，看哪些出现在用户消息中。长关键词命中额外加分，按总分降序排列返回候选列表。这个过程是纯字符串匹配，零 LLM 成本。

**完整加载（带缓存）**：匹配成功后，读取完整的 SKILL.md 文件，去掉 frontmatter，把正文包装为 LLM 友好的格式注入 system prompt。已加载过的 skill 会被缓存，同一会话中不会重复读取文件。

### src/skill_executor.py — 技能执行引擎

负责根据 SKILL.md 中定义的流程实际执行脚本命令。

**命令提取**：从 LLM 的回复文本中提取可执行命令。优先从 bash 代码块（\`\`\`bash ... \`\`\`）中提取，也识别普通行中以 python/node/bun/npx 开头的命令。

**路径解析**：LLM 输出的命令中路径可能是相对于 skill 根目录的相对路径，也可能是相对于项目根目录的路径。解析器会尝试多种策略：先相对于 skill_base_dir 查找，再相对于项目根目录查找，最后在 skill 子目录中按文件名搜索。找到后用绝对路径替换，包含空格或特殊字符（&等）的路径自动用 shlex.quote 加引号。

**Python 解释器适配**：在 conda 环境下，`sh` 可能找不到 `python` 命令。执行引擎自动把命令开头的 `python` 替换为当前 Python 解释器的绝对路径（sys.executable）。

**SVG 提取**：专门为 baoyu-diagram 设计的回退机制。LLM 在回复中内嵌了完整的 SVG 代码（通常包在 cat heredoc 里），但 cat 命令不会被提取和执行。`extract_and_save_svg` 方法从回复文本中直接匹配 `<svg>...</svg>` 标签，提取后保存到 `diagram/<标题>/diagram.svg` 路径。


## 修改文件

### src/agent.py — CLI 版集成

**启动阶段**：初始化 SkillLoader 并调用 discover() 完成 Phase 1，在执行器初始化后打印已发现的 skill 列表。

**对话循环中的 skill 匹配**：每条用户输入在进入 LLM 之前，先做 Phase 2 关键词匹配。如果匹配到 skill，立即触发 Phase 3 加载完整 SKILL.md，并打印匹配信息（skill 名称、加载字符数、备选 skill）。

**System prompt 组装**：在原有的四层记忆基础上，追加两个 skill 相关的内容：匹配到的 skill 完整指令，以及所有可用 skill 的轻量摘要（让 LLM 知道有哪些技能可用）。

**三层脚本执行**：LLM 回复后，分三层尝试执行脚本。Layer 1 从回复中提取命令并执行。Layer 2 在提取为空时按 skill 类型自动构造命令（flash-card 从用户输入提取英文单词→检查 data JSON 是否存在→拼命令）。Layer 3 在 command 拼接和脚本执行都失败时，为 baoyu-diagram 直接从回复提取 SVG 存盘。

**新增 `/skills` 命令**：用户输入 `/skills` 可查看所有已发现 skill 的详细信息（名称、描述、触发关键词列表）。

### src/serve.py — Web 版集成

**Lifespan 初始化**：服务启动时初始化 SkillLoader、调用 discover()、初始化 SkillExecutor，与四层记忆系统同时完成。

**`/chat` 接口增强**：在 SSE 事件流中新增 `skill_matched` 事件（匹配到 skill 时推送）和 `skill_execution` 事件（脚本执行完成时推送）。Skill 完整指令和轻量摘要都注入到 system prompt 中。脚本执行层同样包含三层的回退逻辑。

**新增 API**：`GET /skills` 返回所有已发现 skill 的元信息和触发词。`POST /skills/execute` 允许手动执行指定 skill 的脚本命令。

### memory/AGENTS.md — 技能系统能力声明

新增"技能系统"章节，向 LLM 声明自己拥有可扩展技能系统，说明渐进式加载的工作方式，列出当前可用的两个 skill 及其功能简介。

### requirements.txt — 依赖补充

新增 pyyaml>=6.0（解析 SKILL.md 的 YAML frontmatter）。

### skills/flash-card/SKILL.md — 路径修正

修正脚本路径（移除不存在的 .cursor/ 前缀）。
