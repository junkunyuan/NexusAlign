---
paths:
  - "README.md"
---

# README

Follow these conventions to keep the top-level `README.md` short and current.

1. **Keep only these sections:** what the project is, a table of contents, Updates,
   Algorithms, Installation, Quick Start, and Citation. Keep the table of contents the
   same as the `##` sections, with the same emoji.
2. **Updates:** one dated line per user-facing milestone, newest first, such as a newly
   reproduced paper or a new feature. Write the date as `**[YYYY-MM-DD]**` in US Pacific
   time.
3. **Algorithms:** each reproduced paper, linked to its README under `reproduce/`,
   with no result numbers. Put Paradigms first and Post-Training last, and order the
   papers of each category newest first. `Released` is the date of the first arXiv
   version, as `YYYY-MM-DD`. If a paper has no arXiv version, use the date when it
   first became public, for example in the proceedings. Each domain has its own `###`
   heading and table (for example, Visual Generation). The category does not repeat
   the domain (for example, Post-Training). When we reproduce a new paper, add a row.
   Keep the tables in `reproduce/README.md` the same.
4. **Quick Start:** one short train, evaluate, and kill example (MeanFlow-B/2).
5. **Edit only when needed.** Update the README only when a user-facing change makes it
   wrong or incomplete, for example a new dataset, model, algorithm, or command. Never
   add implementation details, changelogs, or per-experiment notes.
6. **Edit an existing line if you can,** and do not add a new section.
7. **Keep each date and category on one line.** Put a word joiner (U+2060) on each
   side of each hyphen in a date or a category, for example `Post-Training`. Then a
   narrow table does not break it. The joiner is invisible, so add it with code, for
   example in Python: `"\u2060-\u2060".join(text.split("-"))`.
