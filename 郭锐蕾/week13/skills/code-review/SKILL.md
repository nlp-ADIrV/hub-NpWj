---
name: code-review
description: 代码审查。当用户要求 review code、审查代码、code review、检查 PR 时触发。
trigger: code review | review code | 审查代码 | 检查 PR | lint
version: 1.0.0
---

# Code Review Skill（教学演示）

## Steps

1. Read changed files
2. Check: style, logic, security
3. Retrieve relevant docs (RAG，可选)
4. Call lint tool (Function Call / MCP)
5. Return structured report

## Report Format

```markdown
## Summary
## Findings
### Critical
### Major
### Nit
## Suggestions
```

## QA

- [ ] 是否覆盖安全相关检查
- [ ] 是否给出可执行修改建议
- [ ] 是否区分严重级别
