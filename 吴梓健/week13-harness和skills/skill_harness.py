"""
Progressive Skill Loading Harness (通用工具版)
=================================================
通用 skill harness，核心特性：**渐进式技能加载** + **通用工具**。

与 Cursor 类似：harness 不为每个 skill 预写工具函数，而是提供一组通用工具
（write_file / read_file / run_command / list_directory / open_in_browser），
LLM 读完 SKILL.md 后自行决定如何使用这些通用工具来完成任务。

工作原理:
  1. 启动时扫描 skills/ 目录，只解析每个 SKILL.md 的 YAML frontmatter（name + description）
  2. system prompt 中只包含技能摘要（轻量），告知大模型有哪些技能可用
  3. 用户提问 → 大模型根据摘要判断需要哪个技能 → 调用 load_skill 加载完整说明
  4. 完整 SKILL.md + 技能目录路径注入 system prompt；通用工具变为可用
  5. 大模型按 SKILL.md 指令，自主使用通用工具执行任务
  6. 任务完成后调用 release_skill 释放 → 回到轻量状态

新增技能只需把技能目录放到 skills/ 下，无需修改 harness 代码。

用法:
    python skill_harness.py
"""
import json
import locale
import os
import platform
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ===========================================================================
# 路径常量
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent          # homework/
SKILLS_DIR = BASE_DIR / "skills"                    # skills/
ENV_PATH = BASE_DIR / ".env"

# ===========================================================================
# 终端颜色
# ===========================================================================
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def cprint(msg: str, color: str = "", end: str = "\n"):
    """带颜色的 print（兼容 Windows GBK 控制台）"""
    full = f"{color}{msg}{C.RESET}"
    try:
        print(full, end=end)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        safe = full.encode(enc, errors="replace").decode(enc, errors="replace")
        print(safe, end=end)


if sys.platform == "win32":
    os.system("")  # 启用 ANSI 转义码

# ===========================================================================
# 环境变量
# ===========================================================================
load_dotenv(ENV_PATH)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()


def check_env():
    missing = []
    if not LLM_BASE_URL:
        missing.append("LLM_BASE_URL")
    if not LLM_API_KEY:
        missing.append("LLM_API_KEY")
    if missing:
        cprint("\n[错误] 缺少环境变量: " + ", ".join(missing), C.RED)
        cprint(f"请在 {ENV_PATH} 中配置（参考 .env.example）\n", C.YELLOW)
        sys.exit(1)


# ===========================================================================
# YAML Frontmatter 解析器
# ===========================================================================
def parse_frontmatter(text: str) -> tuple[dict, str]:
    """从 Markdown 中解析 YAML frontmatter（name / description / version）。"""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text
    fm, body = match.group(1), match.group(2)
    metadata: dict[str, str] = {}
    lines = fm.split("\n")
    i = 0
    while i < len(lines):
        kv = re.match(r"^(\w[\w-]*)\s*:\s*(.*)", lines[i])
        if kv:
            key, value = kv.group(1), kv.group(2).strip()
            if value in (">-", ">", "|", "|-"):
                block: list[str] = []
                i += 1
                while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                    block.append(lines[i].strip())
                    i += 1
                metadata[key] = " ".join(block)
            elif value:
                metadata[key] = value
                i += 1
            else:
                i += 1
        else:
            i += 1
    return metadata, body


# ===========================================================================
# 通用工具处理函数 — 不绑定任何特定技能
# ===========================================================================
MAX_OUTPUT = 80000  # run_command 输出截断上限（字符）


def tool_write_file(args: dict) -> str:
    """将内容写入文件，自动创建父目录。"""
    file_path = Path(args["file_path"])
    content = args.get("content", "")
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        cprint(f"    [write_file] {file_path} ({len(content)} chars)", C.GREEN)
        return json.dumps({"success": True, "message": f"文件已写入: {file_path}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "message": f"写入失败: {e}"}, ensure_ascii=False)


def tool_read_file(args: dict) -> str:
    """读取文件内容。"""
    file_path = Path(args["file_path"])
    if not file_path.exists():
        return json.dumps({"success": False, "message": f"文件不存在: {file_path}"}, ensure_ascii=False)
    try:
        content = file_path.read_text(encoding="utf-8")
        cprint(f"    [read_file] {file_path} ({len(content)} chars)", C.CYAN)
        return json.dumps({"success": True, "content": content}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "message": f"读取失败: {e}"}, ensure_ascii=False)


# ---- 危险命令拦截 ----
DANGEROUS_PATTERNS = [
    # 系统破坏
    r"rm\s+-rf\s+/",                    # rm -rf /
    r"rmdir\s+/s\s+/q",                 # Windows: rmdir /s /q
    r"del\s+/[fsq].*:\\",              # Windows: del /f /s C:\
    r"format\s+[a-z]:",                 # format C:
    r"mkfs\.",                          # mkfs.ext4 /dev/sda
    r"dd\s+.*if=/dev/(zero|random|null)",  # dd if=/dev/zero
    r"diskpart",                        # Windows diskpart
    # 关机/重启
    r"\bshutdown\b",                   # shutdown
    r"\breboot\b",                     # reboot
    r"\bhalt\b",                       # halt
    r"\bpoweroff\b",                   # poweroff
    # 权限提升
    r"\bsudo\b",                       # sudo
    r"\bsu\s+",                        # su root
    r"\brunas\b",                      # Windows: runas
    # 进程批量终止
    r"\bkill\s+-9\s+-\d",            # kill -9 -1 (kill all)
    r"\bkillall\b",                    # killall
    r"taskkill\s+/[f].*/[t].*\*",      # taskkill /f /im *
    # 注册表/系统配置删除
    r"reg\s+delete",                   # reg delete
    r"regedit\s+/s",                    # regedit /s
    # 管道执行远程脚本（高风险）
    r"\b(curl|wget)\b.*\|\s*(sh|bash|python|perl|ruby)",  # curl ... | sh
    r"\|\s*(sh|bash)\b",              # | sh / | bash
    # 覆盖系统关键文件
    r">\s*/etc/(passwd|shadow|sudoers)",
    r">\s*/boot/",                     # 覆盖 /boot/
    # 网络监听/后门
    r"nc\s+.*-l\s+-p",                # nc -l -p (netcat listen)
    r"\bchmod\s+777\s+/(s?bin|etc|usr|boot|root)",  # chmod 777 /bin
]
DANGEROUS_RE = [re.compile(p, re.IGNORECASE) for p in DANGEROUS_PATTERNS]


def is_dangerous_command(command: str) -> str | None:
    """检查命令是否危险。返回匹配到的危险模式描述，无危险则返回 None。"""
    for pattern in DANGEROUS_RE:
        if pattern.search(command):
            return pattern.pattern
    return None


def tool_run_command(args: dict) -> str:
    """执行 shell 命令，返回 stdout / stderr / returncode。
    内置危险命令拦截：rm -rf /、format、sudo、curl|sh 等将被拒绝。"""
    command = args["command"]
    cwd = args.get("cwd") or str(BASE_DIR)
    timeout = args.get("timeout", 120)

    # 危险命令拦截
    matched = is_dangerous_command(command)
    if matched:
        cprint(f"    [run_command] 拒绝危险命令: {command}", C.RED + C.BOLD)
        cprint(f"    [run_command] 匹配危险模式: {matched}", C.RED)
        return json.dumps({
            "success": False,
            "message": (
                f"命令被拒绝：检测到危险模式 '{matched}'。"
                "出于安全考虑，以下类型的命令被禁止执行："
        "系统删除(rm -rf /)、格式化(format)、权限提升(sudo/su)、"
        "关机重启(shutdown/reboot)、管道执行远程脚本(curl|sh)等。"
        "请使用安全的替代命令。"
            ),
        }, ensure_ascii=False)

    cprint(f"    [run_command] {command}", C.YELLOW)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            cwd=cwd,
            timeout=timeout,
        )
        enc = locale.getpreferredencoding(False) or "utf-8"
        stdout = result.stdout.decode(enc, errors="replace") if result.stdout else ""
        stderr = result.stderr.decode(enc, errors="replace") if result.stderr else ""
        # 截断过长输出
        if len(stdout) > MAX_OUTPUT:
            stdout = stdout[:MAX_OUTPUT] + f"\n... (截断，共 {len(stdout)} 字符)"
        if len(stderr) > MAX_OUTPUT:
            stderr = stderr[:MAX_OUTPUT] + f"\n... (截断，共 {len(stderr)} 字符)"
        if result.returncode == 0 and stdout:
            cprint(f"    [run_command] stdout: {stdout}...", C.DIM)
        elif result.returncode != 0:
            cprint(f"    [run_command] 退出码 {result.returncode}", C.RED)
        return json.dumps({
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "message": f"命令超时（{timeout}秒）"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "message": f"执行失败: {e}"}, ensure_ascii=False)


def tool_list_directory(args: dict) -> str:
    """列出目录内容。"""
    dir_path = Path(args["dir_path"])
    if not dir_path.exists():
        return json.dumps({"success": False, "message": f"目录不存在: {dir_path}"}, ensure_ascii=False)
    items = []
    for item in sorted(dir_path.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        items.append({"name": item.name, "type": "dir" if item.is_dir() else "file"})
    cprint(f"    [list_directory] {dir_path} ({len(items)} items)", C.CYAN)
    return json.dumps({"success": True, "items": items}, ensure_ascii=False)


def tool_open_in_browser(args: dict) -> str:
    """在默认浏览器中打开文件。"""
    file_path = Path(args["file_path"])
    if not file_path.exists():
        return json.dumps({"success": False, "message": f"文件不存在: {file_path}"}, ensure_ascii=False)
    cprint(f"    [open_in_browser] {file_path}", C.MAGENTA)
    webbrowser.open(file_path.resolve().as_uri())
    return json.dumps({"success": True, "message": f"已在浏览器中打开: {file_path}"}, ensure_ascii=False)


# ===========================================================================
# 通用工具定义（OpenAI function-calling 格式）
# ===========================================================================
GENERIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入文件。自动创建不存在的父目录。用于保存 JSON 数据、SVG 文档、代码等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的完整内容"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容。用于读取参考文档、已有数据文件、配置等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "执行 shell 命令。用于运行脚本、安装依赖、执行转换等。返回 stdout/stderr/returncode。\n"
                "安全约束：禁止执行危险命令，包括但不限于：\n"
                "- 系统删除：rm -rf /、rmdir /s /q、del /f /s C:\\*、format、mkfs、dd if=/dev/zero\n"
                "- 关机重启：shutdown、reboot、halt、poweroff\n"
                "- 权限提升：sudo、su、runas\n"
                "- 进程终止：kill -9 -1、killall、taskkill /f /im *\n"
                "- 远程脚本管道：curl ... | sh、wget ... | bash\n"
                "- 注册表/系统文件修改：reg delete、覆盖 /etc/passwd 或 /boot/\n"
                "请仅执行与当前技能任务相关的安全命令。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "cwd": {"type": "string", "description": "工作目录（默认为 harness 目录）"},
                    "timeout": {"type": "integer", "description": "超时秒数（默认 120）"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出目录中的文件和子目录。用于探索技能目录结构。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {"type": "string", "description": "目录路径"},
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_in_browser",
            "description": "在默认浏览器中打开文件。用于预览生成的 HTML/SVG 等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "文件路径"},
                },
                "required": ["file_path"],
            },
        },
    },
]

# 工具名 → 处理函数
GENERIC_HANDLERS = {
    "write_file": tool_write_file,
    "read_file": tool_read_file,
    "run_command": tool_run_command,
    "list_directory": tool_list_directory,
    "open_in_browser": tool_open_in_browser,
}


# ===========================================================================
# Skill 类 — 渐进式加载的核心
# ===========================================================================
class Skill:
    """表示一个技能。启动时只解析 frontmatter（轻量），完整内容延迟加载。"""

    def __init__(self, skill_dir: Path):
        self.skill_dir = skill_dir
        self.skill_md = skill_dir / "SKILL.md"
        self.name = ""
        self.description = ""
        self.version = ""
        self._full_content: str | None = None

        text = self.skill_md.read_text(encoding="utf-8")
        metadata, _body = parse_frontmatter(text)
        self.name = metadata.get("name", skill_dir.name)
        self.description = metadata.get("description", "")
        self.version = metadata.get("version", "")

    @property
    def full_content(self) -> str:
        """延迟加载完整 SKILL.md"""
        if self._full_content is None:
            self._full_content = self.skill_md.read_text(encoding="utf-8")
        return self._full_content

    def get_summary(self) -> str:
        """技能摘要（仅 name + description）"""
        lines = [f"### {self.name}"]
        if self.version:
            lines.append(f"(v{self.version})")
        lines.append(self.description)
        return "\n".join(lines)


# ===========================================================================
# Harness 主类
# ===========================================================================
class Harness:
    """渐进式技能加载 harness（通用工具版）"""

    MAX_LOADED = 2

    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self.loaded_skills: set[str] = set()
        self._scan_skills()

    def _scan_skills(self):
        """扫描 skills/ 目录"""
        if not SKILLS_DIR.exists():
            cprint(f"[警告] skills 目录不存在: {SKILLS_DIR}", C.YELLOW)
            return
        for item in sorted(SKILLS_DIR.iterdir()):
            if item.is_dir() and (item / "SKILL.md").exists():
                try:
                    skill = Skill(item)
                    self.skills[skill.name] = skill
                    cprint(f"  发现技能: {skill.name}", C.DIM)
                except Exception as e:
                    cprint(f"  [跳过] {item.name}: {e}", C.RED)

    # -----------------------------------------------------------------------
    # System Prompt — 每次 API 调用前重建
    # -----------------------------------------------------------------------
    def build_system_prompt(self) -> str:
        parts = [
            "你是一个智能助手，能够根据用户需求自动加载和使用各种技能（skill）。",
            "",
            "## 可用技能摘要",
            "以下是当前可用的技能。当用户的请求匹配某个技能时，调用 `load_skill` 加载完整说明。",
            "对于不涉及技能的普通对话，直接回答即可。",
            "",
        ]
        for i, skill in enumerate(self.skills.values(), 1):
            parts.append(skill.get_summary())
            parts.append("")

        parts.extend([
            "## 通用工具",
            "当技能加载后，以下通用工具变为可用。你可以根据技能说明自主决定如何使用它们：",
            "- **write_file**(file_path, content): 写文件（自动创建父目录）——保存 JSON、SVG、代码等",
            "- **read_file**(file_path): 读文件——读取参考文档、数据文件等",
            "- **run_command**(command, cwd?): 执行 shell 命令——运行脚本、安装依赖等",
            "- **list_directory**(dir_path): 列出目录——探索技能目录结构",
            "- **open_in_browser**(file_path): 浏览器打开文件——预览结果",
            "",
            "## 使用流程",
            "1. 根据用户请求判断需要哪个技能 → 调用 `load_skill`",
            "2. 阅读加载的技能完整说明（SKILL.md）",
            f"3. 按技能说明使用通用工具执行操作（同时最多 {self.MAX_LOADED} 个技能）",
            "4. 任务完成后调用 `release_skill` 释放技能",
            "",
            "## 运行环境",
            f"- 操作系统: {platform.system()} {platform.release()}",
            f"- Python: {sys.executable}",
            f"- Harness 目录: {BASE_DIR}",
            f"- Shell: {'cmd/powershell' if sys.platform == 'win32' else 'bash/sh'}",
            "",
        ])

        # 已加载技能的完整说明
        if self.loaded_skills:
            parts.append("## 已加载技能完整说明")
            parts.append("")
            for name in self.loaded_skills:
                skill = self.skills[name]
                parts.append(f"{'=' * 50}")
                parts.append(f"### 技能: {skill.name}")
                parts.append(f"技能目录（绝对路径）: {skill.skill_dir}")
                parts.append("")
                parts.append(skill.full_content)
                parts.append("")
                parts.append(
                    "重要提示: 上述技能说明中提到的路径请做如下替换——"
                    f"`.cursor/skills/{name}/` 或 `{{baseDir}}` → `{skill.skill_dir}`；"
                    f"`{{projectDir}}` 或 `{{inputFileDir}}` → `{BASE_DIR}`。"
                    f"运行 Python 脚本时请使用 `{sys.executable}` 代替 `python`。"
                )
                parts.append("")
        else:
            parts.append("## 已加载技能")
            parts.append("（当前无技能加载。请根据用户需求调用 load_skill。）")
            parts.append("")

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # 动态工具列表
    # -----------------------------------------------------------------------
    def get_available_tools(self) -> list:
        """系统工具（始终可用）+ 通用工具（有技能加载时可用）"""
        tools = self._get_system_tools()
        if self.loaded_skills:
            tools.extend(GENERIC_TOOLS)
        return tools

    def _get_system_tools(self) -> list:
        skill_names = list(self.skills.keys())
        loaded_names = list(self.loaded_skills)
        return [
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": "加载技能的完整说明到上下文，并激活通用工具。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "enum": skill_names,
                                "description": "要加载的技能名称",
                            },
                        },
                        "required": ["skill_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "release_skill",
                    "description": "释放已加载的技能，从上下文中移除其完整说明。任务完成后应调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "enum": loaded_names,
                                "description": "要释放的技能名称",
                            },
                        },
                        "required": ["skill_name"],
                    },
                },
            },
        ]

    # -----------------------------------------------------------------------
    # 工具执行分发
    # -----------------------------------------------------------------------
    def execute_tool(self, func_name: str, args: dict) -> str:
        # 系统工具
        if func_name == "load_skill":
            return self._handle_load_skill(args)
        if func_name == "release_skill":
            return self._handle_release_skill(args)
        # 通用工具（仅当有技能加载时可用）
        if self.loaded_skills and func_name in GENERIC_HANDLERS:
            return GENERIC_HANDLERS[func_name](args)
        return json.dumps(
            {"success": False, "message": f"未知或不可用的工具: {func_name}"},
            ensure_ascii=False,
        )

    def _handle_load_skill(self, args: dict) -> str:
        name = args.get("skill_name", "").strip()
        if name not in self.skills:
            return json.dumps(
                {"success": False, "message": f"技能 '{name}' 不存在。可用: {list(self.skills.keys())}"},
                ensure_ascii=False,
            )
        if name in self.loaded_skills:
            return json.dumps(
                {"success": True, "message": f"技能 '{name}' 已加载，无需重复加载。"},
                ensure_ascii=False,
            )
        if len(self.loaded_skills) >= self.MAX_LOADED:
            return json.dumps(
                {"success": False,
                 "message": f"已加载 {self.MAX_LOADED} 个技能，请先 release。当前: {list(self.loaded_skills)}"},
                ensure_ascii=False,
            )
        self.loaded_skills.add(name)
        _ = self.skills[name].full_content  # 触发延迟加载
        cprint(f"  >>> 技能已加载: {name}", C.GREEN + C.BOLD)
        cprint(f"      通用工具已激活: {list(GENERIC_HANDLERS.keys())}", C.DIM)
        return json.dumps(
            {"success": True,
             "message": f"技能 '{name}' 已加载。完整说明已注入上下文。通用工具已可用: {list(GENERIC_HANDLERS.keys())}。请阅读技能说明并开始执行。"},
            ensure_ascii=False,
        )

    def _handle_release_skill(self, args: dict) -> str:
        name = args.get("skill_name", "").strip()
        if name not in self.loaded_skills:
            return json.dumps(
                {"success": False, "message": f"技能 '{name}' 未加载。当前: {list(self.loaded_skills)}"},
                ensure_ascii=False,
            )
        self.loaded_skills.discard(name)
        cprint(f"  <<< 技能已释放: {name}", C.YELLOW)
        return json.dumps(
            {"success": True, "message": f"技能 '{name}' 已释放，上下文已回收。"},
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------------
    # 主循环
    # -----------------------------------------------------------------------
    def run(self):
        check_env()

        cprint("=" * 60, C.GRAY)
        cprint("  Progressive Skill Loading Harness", C.BOLD + C.CYAN)
        cprint("  渐进式技能加载 · 通用工具 · 多轮对话", C.GRAY)
        cprint("=" * 60, C.GRAY)
        cprint(f"  模型: {LLM_MODEL}", C.DIM)
        cprint(f"  Base URL: {LLM_BASE_URL}", C.DIM)
        cprint(f"  已发现技能: {list(self.skills.keys())}", C.DIM)
        cprint(f"  通用工具: {list(GENERIC_HANDLERS.keys())}", C.DIM)
        cprint("=" * 60, C.GRAY)
        cprint("  输入 quit/exit 退出", C.GRAY)
        print()

        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        conversation: list[dict] = []

        while True:
            try:
                user_input = input(f"{C.BOLD}{C.GREEN}你>{C.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                cprint("\n再见！", C.YELLOW)
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                cprint("再见！", C.YELLOW)
                break

            conversation.append({"role": "user", "content": user_input})

            # --- 与大模型交互（可能多轮 tool-calling） ---
            cprint(f"{C.BOLD}{C.BLUE}AI>{C.RESET} ", end="")
            assistant_responded = False

            while True:
                # 每次 API 调用前重建 system prompt 和 tools
                messages = [{"role": "system", "content": self.build_system_prompt()}]
                messages.extend(conversation)
                tools = self.get_available_tools()

                # ---- 流式输出 ----
                content_parts: list[str] = []
                tool_calls_map: dict[int, dict] = {}
                tool_calls_started = False

                try:
                    stream = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        stream=True,
                    )
                    for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if not delta:
                            continue

                        # 文本内容 — 实时打印
                        if delta.content:
                            print(delta.content, end="", flush=True)
                            content_parts.append(delta.content)

                        # 工具调用 — 累积参数
                        if delta.tool_calls:
                            if not tool_calls_started:
                                tool_calls_started = True
                                if content_parts and not "".join(content_parts).endswith("\n"):
                                    print()  # 文本后换行（仅当文本未以换行结尾时）
                                cprint("  [正在生成工具调用", C.GRAY, end="")
                                sys.stdout.flush()
                            for tc_delta in delta.tool_calls:
                                idx = tc_delta.index
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                                if tc_delta.id:
                                    tool_calls_map[idx]["id"] = tc_delta.id
                                if tc_delta.function:
                                    if tc_delta.function.name:
                                        tool_calls_map[idx]["name"] = tc_delta.function.name
                                        cprint(f" {tc_delta.function.name}", C.GRAY, end="")
                                    if tc_delta.function.arguments:
                                        tool_calls_map[idx]["arguments"] += tc_delta.function.arguments
                                        # 每 2000 字符打印一个进度点
                                        prev_len = len(tool_calls_map[idx]["arguments"]) - len(tc_delta.function.arguments)
                                        if len(tool_calls_map[idx]["arguments"]) // 2000 > prev_len // 2000:
                                            cprint(".", C.GRAY, end="")
                                            sys.stdout.flush()
                except Exception as e:
                    if tool_calls_started:
                        print()
                    cprint(f"\n[API 错误] {e}", C.RED)
                    break

                # 流结束，收尾
                if tool_calls_started:
                    cprint("]", C.GRAY)

                content = "".join(content_parts) if content_parts else None

                # 重建 tool_calls 列表
                tool_calls = []
                for idx in sorted(tool_calls_map.keys()):
                    tc = tool_calls_map[idx]
                    tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    })

                if content:
                    assistant_responded = True

                if tool_calls:
                    assistant_msg: dict = {"role": "assistant"}
                    if content:
                        assistant_msg["content"] = content
                    assistant_msg["tool_calls"] = tool_calls
                    conversation.append(assistant_msg)

                    for tc in tool_calls:
                        func_name = tc["function"]["name"]
                        cprint(f"  [调用工具] {func_name}", C.YELLOW)
                        try:
                            func_args = json.loads(tc["function"]["arguments"])
                        except json.JSONDecodeError:
                            func_args = {}

                        result = self.execute_tool(func_name, func_args)
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })

                    cprint(f"{C.BOLD}{C.BLUE}AI>{C.RESET} ", end="")
                    continue

                if not assistant_responded:
                    cprint("(空回复)", C.GRAY)
                conversation.append({"role": "assistant", "content": content or ""})
                break

            print()
            if self.loaded_skills:
                cprint(f"  (当前已加载技能: {list(self.loaded_skills)})", C.GRAY)


# ===========================================================================
# 入口
# ===========================================================================
if __name__ == "__main__":
    Harness().run()
