# `.agents/` — Agent rules and skills

Guidance for AI coding agents in this repo. It adds to [`AGENTS.md`](../AGENTS.md).

## `rules/`

The `common-*.md` rules do not depend on this repo, so other projects can copy them
unchanged. Each other rule adds to the common rule of the same topic.

| Rule | Applies to | Key point |
|---|---|---|
| [common-workflow](rules/common-workflow.md) | everywhere | understand the request and ask early, explore first, keep the code simple, follow the existing style, keep the comments current, ask before changing a common rule |
| [common-writing](rules/common-writing.md) | everywhere | ASD-STE100 Simplified Technical English: simple words, short sentences, active voice, short noun clusters |
| [common-code-style](rules/common-code-style.md) | everywhere | English, line length 88, naming, emoji print messages, override order, equation citations, docstrings |
| [common-markdown](rules/common-markdown.md) | `**/*.md` | bold leads, paper citations, HTML tables, dates on one line |
| [common-readme](rules/common-readme.md) | `README.md` | the sections, Updates lines, and edits only for user-facing changes |
| [code-style](rules/code-style.md) | everywhere | registry keys and run names, component and counter order, short script comments |
| [configs](rules/configs.md) | `src/nexus_align/configs/**`, `src/nexus_align/algorithms/**` | algorithm configs mirror `__init__`; every arg has a trailing comment; reproduce configs stay minimal and copy each arg's comment; 8 to 16 checkpoints per run |
| [reproduce](rules/reproduce.md) | `reproduce/**` | what a run record holds, how a paper README reports results, and how the code cites its sources |
| [readme](rules/readme.md) | `README.md` | the Algorithms table and the Quick Start example |

## `skills/`

| Skill | Use for |
|---|---|
| [add-algorithm](skills/add-algorithm/SKILL.md) | how to add an algorithm from a GitHub link, from the paper to a smoke run |

## Other agent tools

The rule and skill files are here, so add content only here. Codex and other agents
start from [`AGENTS.md`](../AGENTS.md). Claude Code reads `.claude/`, which links here:

```
.claude/rules  -> ../.agents/rules
.claude/skills -> ../.agents/skills
```
