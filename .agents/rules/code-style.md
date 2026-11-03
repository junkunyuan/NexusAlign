---
paths:
  - "**/*"
---

# Code Style

Follow these conventions in every file of the repo.

1. **Write in English.** Write all comments and docs in Simplified Technical English,
   as [writing](writing.md) describes.
2. **Keep lines within 88 characters,** the `ruff` limit.
3. **Name things consistently.** `-` joins fields, and `_` joins the words inside a
   field. Registry keys and config `name` values are snake_case (`imagenet_1k`,
   `meanflowdit_b_2`), and timestamps are `%Y%m%d_%H%M%S`. An example run directory is
   `imagenet_1k-meanflowdit_b_2-meanflow-20260903_180000`.
4. **Start every print message with an emoji.** This covers Python `print` and shell
   `echo`. Pick the emoji by meaning, as the existing code does:
   🚀 running, ✅ success, ⚠️ warning, ⏳ in progress, 💾 saved, 🛑 exit, ❌ error,
   📊 results, ⏱️ time cost, 📚 data, 🌐 network, 📦 wrap, 🖥️ system, 📌 info.
5. **Order the components as datasets, models, algorithms, trainers, evaluators,
   rewards.** Use this order wherever several of them appear together: imports,
   registrations, config groups and fields, arguments, calls, and docs. For example,
   `src/nexus_align/__init__.py` imports `nexus_align.datasets`, then `.models`,
   `.algorithms`, `.trainers`, `.evaluators`, and `.rewards`. A component's own config
   comes first in its arguments, for example `cfg_trainer, train_dataset, model,
   algorithm, reward`.
6. **Order the training counters as epoch, step, total_step.** Use this order wherever
   they appear together: definitions, arguments, calls, dict keys, config keys, and
   docs. For example, `save_ckpt_every_n_epochs` comes before `save_ckpt_every_n_steps`.
7. **Define the overrides in the order of the parent.** `BaseAlgorithm` defines
   `prepare`, `forward`, then `sample`, so every algorithm defines them in that order.
   A helper that overrides nothing can be anywhere, but it is usually next to the
   method that uses it.
8. **Cite a paper equation as `see paper Eq. 10`.** For example,
   `u_target = v_hat - time_diff * dudt  # see paper Eq. 10`. Here, "paper" is the
   paper that the file implements. For any other paper, use its name, for example
   `see MeanFlow Eq. 22`.
9. **Cite a paper in Markdown as `[Name](readme) | *Title* ([Venue Year](arxiv))`.**
   Link the name to the paper's README under `reproduce/` and the venue to the paper,
   as the Algorithms tables do: `[MeanFlow](meanflow) | *Mean Flows for One-step
   Generative Modeling* ([NeurIPS 2025](https://arxiv.org/abs/2505.13447))`. In a
   table, the `|` splits the name and title columns. The paper's own README does not
   show the name: `**Paper:** *Title* ([Venue Year](arxiv))`. Never link the whole
   citation, as in `[*Title* (Venue)](arxiv)`.
10. **Put the quotes of a multi-line docstring on their own lines.** A one-line
    docstring stays `"""Summary."""`. A multi-line one looks like this:
    ```python
    """
    Generate samples with z_r = z_t - (t - r) u(z_t, r, t).

    One step uses r = 0 and t = 1. More steps move t from 1 to 0 in num_steps.
    """
    ```
11. **Start each Markdown paragraph and list item with a bold lead,** a short label or
    sentence, then the details. For example, `**Summary:** Model the mean velocity ...`
    or `1. **Verify.** Run a short smoke training ...`. Headings, tables, code
    blocks, link-only lists such as a table of contents, and the introduction under a
    heading need none.
12. **Keep the comments of `*_train.sh` and `*_eval.sh` short.** The header names the
    paper table and the official settings that the script follows. It also gives one
    usage example. The paper README and the script's error messages cover the rest.
    Comment other lines only where the code alone is not clear.
