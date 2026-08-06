---
name: flash-card
description: >-
  为一个英语单词生成静态 HTML 学习闪卡，包含音标、词性、中文释义、
  3 条中英对照例句和近义词。用户要求 flash card、闪卡或单词卡时使用。
version: 1.0.0
---

# Flash Card 单词闪卡

为目标英语单词生成一张可直接打开的静态 HTML 学习卡片。

## 执行流程

1. 从用户请求中识别一个目标英语单词，并将它小写化。
2. 自行生成如下 JSON 数据：
   - `word`：单词。
   - `phonetic`：音标。
   - `pos`：词性。
   - `definition`：中文释义。
   - `examples`：恰好 3 条，每条包含 `en` 和 `zh`。
   - `synonyms`：4 至 6 个近义词。
3. 调用 `write_artifact`，把 JSON 写到
   `artifacts/flash-card/<word>.json`，其中 `skill_name` 为 `flash-card`。
4. 上一步成功后，调用 `run_skill_script`：
   - `skill_name`: `flash-card`
   - `script`: `scripts/make_flashcard.py`
   - `args`:
     `["artifacts/flash-card/<word>.json", "-o", "artifacts/flash-card/<word>.html"]`
5. 只有脚本返回 `exit_code: 0` 时才报告成功，并把 HTML 路径告诉用户。

## 数据质量

- 例句自然、长度适中，并体现该词的典型用法。
- 中文翻译与英文逐条对应。
- 近义词应贴近当前中文释义，不要机械堆砌。
- JSON 必须是合法 UTF-8 文本，不能包含 Markdown 代码围栏。

## 输出契约

最终只需说明：

- 已生成的单词；
- HTML 文件路径；
- 若脚本失败，说明真实错误，不得虚构产物。
