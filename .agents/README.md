# `.agents/` — Agent rules and skills

Repo-local guidance for AI coding agents, complementing [`AGENTS.md`](../AGENTS.md).

## `rules/`

| Rule | Applies to | Key point |
|---|---|---|
| [code-style](rules/code-style.md) | everywhere | English, line length 88, naming, emoji print messages |
| [configs](rules/configs.md) | `src/nexus_align/configs/**`, `src/nexus_align/algorithms/**` | algorithm configs mirror `__init__`; reproduce configs stay minimal |
| [reproduce](rules/reproduce.md) | `reproduce/**` | what a run record holds and how a paper README reports results |
| [readme](rules/readme.md) | `README.md` | keep it short; edit only for user-facing changes |

## `skills/`

| Skill | Use for |
|---|---|
| [add-algorithm](skills/add-algorithm/SKILL.md) | adding an algorithm from a GitHub link, from the paper to a smoke run |

## Other agent tools

The files live here. `.claude/` links to `rules/` and `skills/` in this directory, so
Claude Code finds the same content at the path it looks for; Codex and other agents
start from [`AGENTS.md`](../AGENTS.md):

```
.claude/rules  -> ../.agents/rules
.claude/skills -> ../.agents/skills
```

Add content only here.
