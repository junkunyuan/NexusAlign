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
3. **Algorithms:** each reproduced paper, newest first, linked to its README under
   `reproduce/`, with no result numbers. `Published` is the date of its first arXiv
   version, as `YYYY-MM-DD`. Each category names the domain too (for example, Visual
   Generation Post-Training). When we reproduce a new paper, add a row. Keep the table
   in `reproduce/README.md` the same.
4. **Quick Start:** one short train, evaluate, and kill example (MeanFlow-B/2).
5. **Edit only when needed.** Update the README only when a user-facing change makes it
   wrong or incomplete, for example a new dataset, model, algorithm, or command. Never
   add implementation details, changelogs, or per-experiment notes.
6. **Edit an existing line if you can,** and do not add a new section.
