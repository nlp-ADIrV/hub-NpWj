"""
Skill 执行器 —— 把 ReAct 的 Action 文本路由到对应脚本并执行

教学重点：
  1. Action 语法：`skill-name(key=value, key2="value2")`
  2. 内置动作 write_file：保存 SVG / JSON 等文本产物（不依赖具体脚本）
  3. Python 脚本双形态执行：
       - 定义了 run(**kwargs)  → 直接 import 调用（函数形态）
       - 只有 main()/CLI       → 生成命令行参数，用 subprocess 跑（CLI 形态）
  4. 所有脚本产物统一落到 outputs/ 目录，便于查看

用法：
  from src.skill_executor import SkillExecutor, parse_action
  ex = SkillExecutor(registry)
  ex.execute('flash-card(data="/path/x.json")')
"""

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from src.skill_registry import SkillRegistry, PROJECT_ROOT

# 所有脚本产物（HTML/SVG/PNG/JSON...）统一输出目录
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Action 格式：skill-name(kw1=v1, kw2="v2")
_NAME_RE = re.compile(r"^\s*([a-zA-Z][\w-]*)\s*\((.*)\)\s*$", re.DOTALL)


def _split_kv(raw: str) -> list[str]:
    """按顶层逗号拆分 key=value 段，带引号（含内部逗号）的内容不会被误切。"""
    parts, buf, quote = [], "", None
    for ch in raw:
        if quote:
            buf += ch
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf += ch
        elif ch == ",":
            parts.append(buf)
            buf = ""
        else:
            buf += ch
    parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


def _unescape(s: str) -> str:
    """
    安全反转义字符串内容：只处理 JSON 常见的反斜杠转义（双引号/反斜杠/换行/制表符/回车/斜杠），
    其它未知转义（如 \\a、\\w，常见于 Windows 路径）原样保留，避免路径被破坏。
    """
    out, i, n = [], 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\", "/": "/"}
            if nxt in mapping:
                out.append(mapping[nxt])
            else:
                out.append(ch)  # 未知转义：保留原样
                out.append(nxt)
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_action(action_text: str) -> tuple[str, dict]:
    """解析 Action 文本 -> (动作名, kwargs dict)"""
    m = _NAME_RE.match(action_text)
    if not m:
        raise ValueError(f"Action 语法错误：{action_text!r}（应为 skill-name(key=value)）")
    name = m.group(1)
    kwargs: dict = {}
    raw = m.group(2).strip()
    for part in _split_kv(raw):
        key, sep, val = part.partition("=")
        if not sep:
            raise ValueError(f"Action 参数缺少 '='：{part!r}")
        key, val = key.strip(), val.strip()
        first = val[:1]
        if first in "\"'":
            # 引号包裹的字符串：去引号并做安全反转义（保留反斜杠路径）
            inner = val[1:-1] if len(val) >= 2 and val[-1] == first else val
            kwargs[key] = _unescape(inner)
        elif first in "[{tfn0123456789":
            # 字面量形态：尝试转成 int/float/bool/None/list/dict
            try:
                kwargs[key] = ast.literal_eval(val)
            except Exception:
                kwargs[key] = val
        else:
            # 其余原样当字符串（如含反斜杠的未加引号路径）
            kwargs[key] = val
    return name, kwargs


def _script_arg_spec(script: Path) -> dict[str, dict]:
    """
    用 AST 读取脚本里的 argparse.add_argument 定义，得到每个 dest 的参数形态：
      {dest: {"long": "--output" 或 None, "short": "-o" 或 None, "positional": True/False}}
    用于把 kwargs 映射成命令行参数（CLI 形态执行）。
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    spec: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"):
            continue
        names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if not names:
            continue
        dest = None
        long = short = None
        positional = True
        for n in names:
            if n.startswith("--"):
                long = n
                dest = n[2:].replace("-", "_")
                positional = False
            elif n.startswith("-") and len(n) == 2:
                short = n
                positional = False
            else:
                dest = n.replace("-", "_")
        if dest:
            spec[dest] = {"long": long, "short": short, "positional": positional}
    return spec


def _build_cli_args(script: Path, kwargs: dict) -> list[str]:
    """按脚本的 argparse 定义，把 kwargs 映射成命令行参数列表"""
    spec = _script_arg_spec(script)
    argv: list[str] = []
    for key, val in kwargs.items():
        meta = spec.get(key)
        text = val if isinstance(val, str) else str(val)
        if meta is None:
            # 脚本没声明该参数，用 --key=value 兜底
            argv.append(f"--{key}={text}")
        elif meta["positional"]:
            argv.append(text)
        elif meta["long"]:
            argv.append(f"{meta['long']}={text}")
        else:
            argv.extend([meta["short"], text])
    return argv


def _strip_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


class SkillExecutor:
    """执行一个 Action：内置动作 或 对应 skill 的脚本。"""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def execute(self, action_text: str) -> dict:
        """
        执行 Action，返回统一结构：
          {action, ok, stdout, stderr, error, outputs}
        outputs: 本次执行产生的文件路径列表
        """
        try:
            name, kwargs = parse_action(action_text)
        except ValueError as e:
            return {"action": action_text, "ok": False, "stdout": "", "stderr": "", "error": str(e), "outputs": []}

        # ── 内置动作：write_file ──────────────────────────────────────
        if name == "write_file":
            return self._do_write_file(kwargs)

        # ── 技能脚本执行 ──────────────────────────────────────────────
        if not self.registry.exists(name):
            return {"action": action_text, "ok": False, "stdout": "", "stderr": "",
                    "error": f"技能不存在：{name}（可用：{self.registry.names()}）", "outputs": []}
        meta = self.registry.meta(name)
        if not meta.has_scripts:
            return {"action": action_text, "ok": False, "stdout": "", "stderr": "",
                    "error": f"技能 {name} 没有可执行脚本（scripts/ 目录不存在），直接给出回答即可。", "outputs": []}

        # 找到第一个可导入的 python 脚本
        script = self._find_python_script(meta.path / "scripts")
        if script is None:
            return {"action": action_text, "ok": False, "stdout": "", "stderr": "",
                    "error": f"技能 {name} 的 scripts/ 下没有 python 脚本，无法自动执行，请按 SKILL.md 说明输出内容。", "outputs": []}

        return self._run_python(script, name, kwargs)

    # ── 内置动作实现 ──────────────────────────────────────────────────
    def _do_write_file(self, kwargs: dict) -> dict:
        path_raw = _strip_quotes(str(kwargs.get("path", kwargs.get("file", ""))))
        content = str(kwargs.get("content", kwargs.get("text", "")))
        if not path_raw:
            return {"action": "write_file", "ok": False, "stdout": "", "stderr": "",
                    "error": "write_file 需要 path 参数", "outputs": []}
        out_path = Path(path_raw)
        if not out_path.is_absolute():
            out_path = OUTPUTS_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        return {"action": "write_file", "ok": True, "stdout": f"已写入 {len(content)} 字符 → {out_path}",
                "stderr": "", "error": "", "outputs": [str(out_path)]}

    # ── 脚本定位与执行 ────────────────────────────────────────────────
    @staticmethod
    def _find_python_script(scripts_dir: Path):
        for f in sorted(scripts_dir.iterdir()):
            if f.suffix == ".py":
                return f
        return None

    def _resolve_file_value(self, val):
        """
        对路径类参数做宽容解析：原样 / 相对 outputs/ / 技能 data 目录。
        命中真实文件就返回绝对路径；若 val 是无扩展名的单词（如 "crazy"），
        还会尝试 outputs/data/<val>.json 等常见数据文件名。
        均不命中则原样返回。
        """
        if not isinstance(val, str):
            return val
        candidates = [Path(val), OUTPUTS_DIR / val]
        for m in self.registry.names():
            candidates.append(self.registry.meta(m).path / "data" / val)
        # 无扩展名的单词 → 尝试常见数据文件
        if "." not in Path(val).name:
            candidates += [OUTPUTS_DIR / "data" / f"{val}.json",
                           OUTPUTS_DIR / f"{val}.json"]
            for m in self.registry.names():
                candidates.append(self.registry.meta(m).path / "data" / f"{val}.json")
        for cand in candidates:
            if cand.is_file():
                return str(cand)
        return val

    @staticmethod
    def _module_has_run(module) -> bool:
        return hasattr(module, "run") and callable(getattr(module, "run"))

    def _run_python(self, script: Path, skill_name: str, kwargs: dict) -> dict:
        # 方案 A：函数形态 —— import 后调用 run(**kwargs)
        if kwargs:
            spec = importlib.util.spec_from_file_location(f"{skill_name}_script", script)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    sys.path.insert(0, str(script.parent))
                    spec.loader.exec_module(module)
                finally:
                    sys.path.pop(0)
                if self._module_has_run(module):
                    try:
                        out = module.run(**kwargs)
                        return {"action": skill_name, "ok": True,
                                "stdout": str(out) if out else "run() 执行完成",
                                "stderr": "", "error": "", "outputs": []}
                    except TypeError as e:
                        # 参数不匹配 → 降级到 CLI 形态
                        pass
        # 方案 B：CLI 形态 —— subprocess 按脚本的 argparse 定义映射参数
        resolved = {k: self._resolve_file_value(v) for k, v in kwargs.items()}
        args = [str(script), *_build_cli_args(script, resolved)]
        try:
            proc = subprocess.run(
                [sys.executable, *args],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=OUTPUTS_DIR, timeout=120,
            )
            return {"action": skill_name, "ok": proc.returncode == 0,
                    "stdout": proc.stdout, "stderr": proc.stderr,
                    "error": "" if proc.returncode == 0 else f"退出码 {proc.returncode}",
                    "outputs": []}
        except subprocess.TimeoutExpired:
            return {"action": skill_name, "ok": False, "stdout": "", "stderr": "",
                    "error": "脚本执行超时（>120s）", "outputs": []}
        except OSError as e:
            return {"action": skill_name, "ok": False, "stdout": "", "stderr": "",
                    "error": f"脚本无法执行：{e}", "outputs": []}
