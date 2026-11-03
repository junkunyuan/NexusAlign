# `.agents/` — Agent rules and skills

Guidance for AI coding agents in this repo. It adds to [`AGENTS.md`](../AGENTS.md).

## `rules/`

| Rule | Applies to | Key point |
|---|---|---|
| [code-style](rules/code-style.md) | everywhere | English, line length 88, naming, emoji print messages, component and counter order, override order, equation and paper citations, docstrings, bold leads in Markdown, short script comments |
| [writing](rules/writing.md) | everywhere | ASD-STE100 Simplified Technical English: simple words, short sentences, active voice, short noun clusters |
| [configs](rules/configs.md) | `src/nexus_align/configs/**`, `src/nexus_align/algorithms/**` | algorithm configs mirror `__init__`; every arg has a trailing comment; reproduce configs stay minimal and copy each arg's comment; 8 to 16 checkpoints per run |
| [reproduce](rules/reproduce.md) | `reproduce/**` | what a run record holds, how a paper README reports results, and how the code cites its sources |
| [readme](rules/readme.md) | `README.md` | keep it short; edit only for user-facing changes |

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
