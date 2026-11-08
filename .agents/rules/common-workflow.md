---
paths:
  - "**/*"
---

# Common Workflow

Follow these steps for every change to the project. A project rule can add to them.

1. **Understand:** When the user gives a request, analyze it before you start. Find
   its goal, its challenges, and its difficult parts.
   - **Ask early:** if something is not clear, or if the user must make a decision,
     ask at once. Ask before the long work, because the user can leave after the
     request.
2. **Explore:** Explore the relevant files and understand the existing code before you
   make changes.
3. **Simplicity:** Always keep the code as simple as possible, when you add a feature
   and when you change code. Reuse an existing helper, and do not copy code. Do not add
   an option or abstraction that nothing uses yet. Remove the code that a change leaves
   unused. Avoid unnecessary changes and unnecessary comments.
4. **Style:** Strictly follow the project's existing code style when you write new code
   and comments.
5. **Comments:** Always keep comments and docstrings up to date with the code. If a
   change makes one wrong (for example, a removed feature, a renamed key, or a new
   default), update it in the same change. This includes the trailing comments in
   configs and the docs.
6. **Common rules:** Do not change the `common-*.md` rules. If you think that a change
   is necessary, ask the user first.
