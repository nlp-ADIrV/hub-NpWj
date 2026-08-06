---
name: pptx
description: PPT 生成与编辑。当用户提到 PPT、幻灯片、.pptx、slides、deck 时触发。
trigger: .pptx | slides | deck | PPT | 幻灯片
version: 1.0.0
---

# PPT Skill（教学演示）

演示「触发后才加载完整定义」的轻量 Skill。

## Steps

1. Reading：`python -m markitdown input.pptx`
2. Creating：使用 pptxgenjs / python-pptx 生成幻灯片
3. QA：`pdftoppm` 导出预览并人工检查

## Output

- 产出 `.pptx` 文件
- 附带预览图与自检清单

## QA

- [ ] 标题层级清晰
- [ ] 单页信息不过载
- [ ] 中英混排字体正常
