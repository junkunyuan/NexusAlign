---
paths:
  - "**/*.md"
---

# Common Markdown

Follow these conventions in every Markdown file of the project. A project rule can add
to them.

1. **Start each Markdown paragraph and list item with a bold lead,** a short label or
   sentence, then the details. For example, `**Summary:** Train a model ...` or
   `1. **Verify.** Run a short smoke training ...`. Headings, tables, code blocks,
   link-only lists such as a table of contents, and the introduction under a heading
   need none.
2. **Cite a paper as `*Title* ([Venue Year](arxiv))`.** Link the venue to the paper,
   for example `*Attention Is All You Need*
   ([NeurIPS 2017](https://arxiv.org/abs/1706.03762))`.
3. **Use an HTML table only if a cell must span columns,** with `colspan`. GitHub
   renders HTML tables like Markdown tables, but it removes `style` and `class`.
   - **Tags:** GitHub does not render Markdown in the cells, so use `<a href>` for a
     link, `<i>` for italics, `<b>` for bold, and `<br>` for a line break.
   - **Lines:** do not put a blank line in the table, because a blank line ends the
     HTML block. If a cell is long, wrap it over more source lines.
4. **Keep each date and each short label on one line.** In an HTML table, give the
   cell the `nowrap` attribute. In other text, put a word joiner (U+2060) on each side
   of each hyphen, for example in `2026-11-04` or `Post-Training`. Then a narrow page
   does not break it. The joiner is invisible, so add it with code, for example in
   Python: `"⁠-⁠".join(text.split("-"))`.
