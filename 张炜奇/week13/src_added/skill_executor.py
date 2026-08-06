"""
Skill Executor — 技能执行引擎

根据 SKILL.md 中定义的执行流程，在受控环境中执行脚本命令。
支持 Python 脚本和 TypeScript (bun/npx) 脚本。

执行策略：
  1. 解析 LLM 输出或 SKILL.md 中的脚本调用指令
  2. 解析相对路径（相对于 skill 根目录），支持 .cursor/skills/ 路径自动映射
  3. 自动用 shlex.quote() 包裹含特殊字符的文件路径
  4. 设置超时和错误捕获
  5. 返回结构化执行结果

使用方式：
  from src.skill_executor import SkillExecutor

  executor = SkillExecutor()
  result = executor.run_script(
      "python scripts/make_flashcard.py data/test.json",
      skill_base_dir=Path("skills/flash-card"),
      cwd=Path.cwd(),
  )
"""

import os
import re
import sys
import shlex
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 脚本命令模式：匹配 python / python3 / node / bun / npx / npm 开头的命令
_SCRIPT_CMD_RE = re.compile(
    r'^(python\d*|node|bun|npx|npm)\s+.+',
    re.MULTILINE,
)
# bash 代码块中的命令
_BASH_BLOCK_RE = re.compile(
    r'```(?:bash|sh|shell)?\s*\n(.*?)```',
    re.DOTALL,
)

DEFAULT_TIMEOUT = 120  # 脚本默认超时秒数


@dataclass
class ExecutionResult:
    """脚本执行结果"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = -1
    command: str = ""
    timeout: bool = False

    def summary(self) -> str:
        status = "✅ 成功" if self.success else f"❌ 失败 (code={self.returncode})"
        lines = [f"[Skill Executor] {status} — {self.command}"]
        if self.stdout:
            lines.append(f"  stdout: {self.stdout[:200]}")
        if self.stderr:
            lines.append(f"  stderr: {self.stderr[:200]}")
        if self.timeout:
            lines.append(f"  ⚠️ 超时（>{DEFAULT_TIMEOUT}秒）")
        return "\n".join(lines)


class SkillExecutor:
    """技能执行引擎

    负责在 skill 目录上下文中安全地执行脚本。
    自动解析 SKILL.md 中的相对路径为绝对路径。
    """

    def __init__(self, default_timeout: int = DEFAULT_TIMEOUT):
        self.default_timeout = default_timeout

    @staticmethod
    def _use_conda_python(command: str) -> str:
        """
        将命令中的 'python' / 'python3' 替换为当前 Python 解释器的绝对路径。
        解决 conda 环境下 sh 找不到 python 的问题。
        """
        parts = command.strip().split()
        if parts and parts[0] in ('python', 'python3'):
            parts[0] = sys.executable
            return ' '.join(parts)
        return command

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def run_script(
        self,
        command: str,
        skill_base_dir: Path,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """
        在 skill 的目录上下文中执行一条命令。

        Args:
            command: shell 命令（如 "python scripts/make_flashcard.py data/test.json"）
            skill_base_dir: skill 的根目录（scripts/ 的父目录）
            cwd: 执行时的工作目录（默认当前目录）
            timeout: 超时秒数（默认 DEFAULT_TIMEOUT）
        """
        timeout = timeout or self.default_timeout
        work_dir = cwd or Path.cwd()

        # 解析命令：将相对路径转为相对于 skill_base_dir 的绝对路径
        resolved_cmd = self._resolve_paths(command, skill_base_dir)

        # 使用当前 conda 环境的 Python（而非系统默认的 python 命令）
        resolved_cmd = self._use_conda_python(resolved_cmd)

        logger.info(f"执行 skill 脚本：{resolved_cmd}")
        logger.info(f"  工作目录：{work_dir}")

        try:
            result = subprocess.run(
                resolved_cmd,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ},  # 继承环境变量（含 API Key）
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                returncode=result.returncode,
                command=resolved_cmd,
            )
        except subprocess.TimeoutExpired:
            logger.error(f"脚本超时（{timeout}秒）：{resolved_cmd}")
            return ExecutionResult(
                success=False,
                stderr=f"命令执行超时（>{timeout}秒）",
                returncode=-1,
                command=resolved_cmd,
                timeout=True,
            )
        except FileNotFoundError as e:
            logger.error(f"命令未找到：{e}")
            return ExecutionResult(
                success=False,
                stderr=f"命令未找到：{e}",
                returncode=-1,
                command=resolved_cmd,
            )
        except Exception as e:
            logger.error(f"执行异常：{e}")
            return ExecutionResult(
                success=False,
                stderr=str(e),
                returncode=-1,
                command=resolved_cmd,
            )

    # ── 命令解析 ──────────────────────────────────────────────────────────────

    @staticmethod
    def extract_commands_from_llm_output(llm_output: str) -> list[str]:
        """
        从 LLM 的输出中提取脚本命令。

        识别来源：
          1. ```bash ... ``` 代码块
          2. 以 python/node/bun/npx 开头的行
        """
        commands: list[str] = []

        # 1. bash 代码块
        for m in _BASH_BLOCK_RE.finditer(llm_output):
            block = m.group(1).strip()
            for line in block.split('\n'):
                line = line.strip()
                if _SCRIPT_CMD_RE.match(line):
                    commands.append(line)

        # 2. 普通行（非代码块内）
        for line in llm_output.split('\n'):
            line = line.strip()
            if _SCRIPT_CMD_RE.match(line) and line not in commands:
                commands.append(line)

        return commands

    @staticmethod
    def extract_commands_from_skill_md(skill_md_content: str) -> list[tuple[str, str]]:
        """
        从 SKILL.md 中提取预定义的脚本命令及其说明。
        返回 [(描述, 命令), ...]
        """
        results: list[tuple[str, str]] = []
        lines = skill_md_content.split('\n')

        for i, line in enumerate(lines):
            stripped = line.strip()
            if _SCRIPT_CMD_RE.match(stripped):
                # 尝试取前一行作为描述
                desc = ""
                if i > 0:
                    prev = lines[i - 1].strip()
                    if prev and not prev.startswith('```') and not prev.startswith('#'):
                        desc = prev[:80]
                results.append((desc, stripped))

        return results

    # ── 路径解析 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_paths(command: str, skill_base_dir: Path) -> str:
        """
        将命令中的相对路径解析为绝对路径。
        直接用实际文件路径，不做 .cursor/ 这种不存在的映射。

        策略（按优先级）：
          1. 相对于 skill_base_dir（如 scripts/xxx.py、data/xxx.json）
          2. 相对于项目根目录（如 skills/flash-card/scripts/xxx.py）
          3. 在 skill 子目录中按文件名搜索
        """
        try:
            parts = shlex.split(command.strip())
        except ValueError:
            parts = command.strip().split()
        resolved: list[str] = []

        project_root = skill_base_dir.parent.parent

        def _needs_quote(s: str) -> bool:
            return bool(set(s) & {' ', '&', '(', ')', ';', '|', '<', '>', '$', '!', '`', "'", '"'})

        for part in parts:
            if part.startswith('-') or '=' in part:
                resolved.append(part)
                continue

            found = None

            # 策略1：相对于 skill_base_dir
            candidate = skill_base_dir / part
            if candidate.exists():
                found = str(candidate)

            # 策略2：相对于项目根目录
            if not found:
                candidate2 = project_root / part
                if candidate2.exists():
                    found = str(candidate2)

            # 策略3：在 skill 子目录中按文件名搜索
            if not found:
                fname = Path(part).name
                if fname != part:
                    for subdir in skill_base_dir.iterdir():
                        if subdir.is_dir() and not subdir.name.startswith('.'):
                            hit = subdir / fname
                            if hit.exists():
                                found = str(hit)
                                break

            if found:
                if _needs_quote(found):
                    resolved.append(shlex.quote(found))
                else:
                    resolved.append(found)
            else:
                resolved.append(part)

        return ' '.join(resolved)

    # ── 便捷方法 ──────────────────────────────────────────────────────────────

    def execute_skill_scripts(
        self,
        skill_md_content: str,
        skill_base_dir: Path,
        cwd: Path | None = None,
    ) -> list[ExecutionResult]:
        """
        扫描 SKILL.md 中所有脚本命令并逐一执行。
        适用于自动化执行 skill 的完整流程。
        """
        commands = self.extract_commands_from_skill_md(skill_md_content)
        results: list[ExecutionResult] = []

        for desc, cmd in commands:
            logger.info(f"执行：{desc} → {cmd}")
            result = self.run_script(cmd, skill_base_dir, cwd=cwd)
            results.append(result)
            if not result.success:
                logger.warning(f"命令失败，停止后续执行：{cmd}")
                break

        return results

    # ── SVG 提取（baoyu-diagram 回退用）────────────────────────────────────────

    @staticmethod
    def extract_and_save_svg(llm_output: str, cwd: Path) -> Path | None:
        """
        从 LLM 回复中提取 <svg>...</svg> 并保存为文件。
        用于 baoyu-diagram：LLM 把 SVG 嵌在回复里（含 cat/heredoc 等无法执行的命令），
        系统自动提取 SVG 内容存盘。

        保存路径：<cwd>/diagram/<从svg内容推断标题>/diagram.svg
        返回保存的文件路径，没有 SVG 则返回 None。
        """
        svg_re = re.compile(r'(<svg\b[^>]*>.*?</svg>)', re.DOTALL | re.IGNORECASE)
        matches = svg_re.findall(llm_output)
        if not matches:
            return None

        svg_content = matches[0]

        # 尝试从第一个大字号 title text 推断文件名
        title_match = re.search(r'<text[^>]*font-size="(?:15|16|18|20)"[^>]*>([^<]+)</text>', svg_content)
        slug = "diagram"
        if title_match:
            title = title_match.group(1).strip()
            slug = re.sub(r'[^\w一-鿿-]', '-', title).strip('-')[:40] or "diagram"

        out_dir = cwd / "diagram" / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        # 确保 viewBox 存在
        if 'viewBox' not in svg_content[:200]:
            svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 800 600"', 1)

        out_path = out_dir / "diagram.svg"
        out_path.write_text(svg_content, encoding="utf-8")
        logger.info(f"SVG 已提取保存：{out_path} ({len(svg_content)} 字符)")
        return out_path
