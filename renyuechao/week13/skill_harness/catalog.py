"""Skill discovery with frontmatter-only startup loading."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_MAX_FRONTMATTER_CHARS = 16_000


class SkillCatalogError(ValueError):
    """Raised when a skill package cannot be discovered or validated."""


@dataclass(frozen=True)
class SkillSummary:
    """The small, always-loaded portion of a skill."""

    name: str
    description: str
    root: Path
    version: str | None = None
    trigger: str | list[str] | None = None

    def prompt_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "name": self.name,
            "description": " ".join(self.description.split()),
        }
        if self.trigger:
            record["trigger"] = self.trigger
        return record


class SkillCatalog:
    """Discover skill metadata without loading any instruction bodies."""

    def __init__(
        self,
        skills_dir: str | Path,
        *,
        max_instruction_chars: int = 80_000,
    ) -> None:
        self.skills_dir = Path(skills_dir).resolve()
        self.max_instruction_chars = max_instruction_chars
        self._skills: dict[str, SkillSummary] = {}
        self._discover()

    def _discover(self) -> None:
        if not self.skills_dir.is_dir():
            raise SkillCatalogError(f"skills directory does not exist: {self.skills_dir}")

        skill_files = sorted(self.skills_dir.glob("*/SKILL.md"))
        if not skill_files:
            raise SkillCatalogError(
                f"no skills found under {self.skills_dir}; expected */SKILL.md"
            )

        for skill_file in skill_files:
            metadata = _read_frontmatter_only(skill_file)
            summary = _build_summary(skill_file, metadata)
            if summary.name in self._skills:
                previous = self._skills[summary.name].root / "SKILL.md"
                raise SkillCatalogError(
                    f"duplicate skill name {summary.name!r}: {previous} and {skill_file}"
                )
            self._skills[summary.name] = summary

    def __len__(self) -> int:
        return len(self._skills)

    def summaries(self) -> tuple[SkillSummary, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def get(self, name: str) -> SkillSummary:
        try:
            return self._skills[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._skills))
            raise SkillCatalogError(
                f"unknown skill {name!r}; available skills: {available}"
            ) from exc

    def prompt_index(self) -> str:
        """Return only compact metadata for the always-loaded system prompt."""

        records = [summary.prompt_record() for summary in self.summaries()]
        return json.dumps(records, ensure_ascii=False, separators=(",", ":"))

    def load_instructions(self, name: str) -> str:
        """Load a selected skill body after routing has chosen it."""

        summary = self.get(name)
        skill_file = summary.root / "SKILL.md"
        text = skill_file.read_text(encoding="utf-8")
        _, body = _split_frontmatter(text, skill_file)
        if len(body) > self.max_instruction_chars:
            raise SkillCatalogError(
                f"{skill_file} is too large ({len(body)} chars, "
                f"limit {self.max_instruction_chars})"
            )
        return body.strip()

    def resource_manifest(self, name: str) -> tuple[str, ...]:
        """List resource names without reading their contents."""

        root = self.get(name).root
        resources: list[str] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "SKILL.md":
                continue
            relative = path.relative_to(root)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            resources.append(relative.as_posix())
        return tuple(resources)


def _read_frontmatter_only(skill_file: Path) -> dict[str, Any]:
    """Read through the closing frontmatter delimiter and stop there."""

    try:
        with skill_file.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
            if first_line.strip() != "---":
                raise SkillCatalogError(
                    f"{skill_file} must start with YAML frontmatter delimiter '---'"
                )

            frontmatter_lines: list[str] = []
            frontmatter_chars = 0
            for line in handle:
                if line.strip() == "---":
                    break
                frontmatter_chars += len(line)
                if frontmatter_chars > _MAX_FRONTMATTER_CHARS:
                    raise SkillCatalogError(
                        f"{skill_file} frontmatter exceeds "
                        f"{_MAX_FRONTMATTER_CHARS} characters"
                    )
                frontmatter_lines.append(line)
            else:
                raise SkillCatalogError(
                    f"{skill_file} is missing the closing frontmatter delimiter"
                )
    except UnicodeDecodeError as exc:
        raise SkillCatalogError(f"{skill_file} is not valid UTF-8") from exc

    try:
        metadata = yaml.safe_load("".join(frontmatter_lines))
    except yaml.YAMLError as exc:
        raise SkillCatalogError(f"invalid YAML frontmatter in {skill_file}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise SkillCatalogError(f"frontmatter in {skill_file} must be a mapping")
    return metadata


def _build_summary(skill_file: Path, metadata: dict[str, Any]) -> SkillSummary:
    name = metadata.get("name")
    description = metadata.get("description")

    if not isinstance(name, str) or not name.strip():
        raise SkillCatalogError(f"{skill_file} frontmatter requires a non-empty name")
    name = name.strip()
    if not _SKILL_NAME_RE.fullmatch(name):
        raise SkillCatalogError(
            f"{skill_file} has invalid skill name {name!r}; "
            "use lowercase letters, digits, '.', '_' or '-'"
        )
    if not isinstance(description, str) or not description.strip():
        raise SkillCatalogError(
            f"{skill_file} frontmatter requires a non-empty description"
        )

    version_value = metadata.get("version")
    version = str(version_value) if version_value is not None else None
    trigger = metadata.get("trigger")
    if trigger is not None and not isinstance(trigger, (str, list)):
        raise SkillCatalogError(
            f"{skill_file} trigger must be a string or a list of strings"
        )
    if isinstance(trigger, list) and not all(isinstance(item, str) for item in trigger):
        raise SkillCatalogError(
            f"{skill_file} trigger list must contain strings only"
        )

    return SkillSummary(
        name=name,
        description=description.strip(),
        root=skill_file.parent.resolve(),
        version=version,
        trigger=trigger,
    )


def _split_frontmatter(text: str, skill_file: Path) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise SkillCatalogError(f"{skill_file} has no valid frontmatter")

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[1:index]), "".join(lines[index + 1 :])
    raise SkillCatalogError(f"{skill_file} is missing the closing frontmatter delimiter")
