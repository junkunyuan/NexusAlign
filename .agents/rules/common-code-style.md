---
paths:
  - "**/*"
---

# Common Code Style

Follow these conventions in every file of the project. A project rule can add to them.

1. **Write in English.** Write all comments and docs in Simplified Technical English,
   as [common-writing](common-writing.md) describes.
2. **Keep lines within 88 characters,** the `ruff` limit.
3. **Name things consistently.** `-` joins fields, and `_` joins the words inside a
   field. Keys and `name` values are snake_case, and timestamps are
   `%Y%m%d_%H%M%S`. An example run directory is
   `imagenet_1k-resnet_50-20260903_180000`.
4. **Start every print message with an emoji.** This covers Python `print` and shell
   `echo`. Pick the emoji by meaning, such as:
   🚀 running, ✅ success, ⚠️ warning, ⏳ in progress, 💾 saved, 🛑 exit, ❌ error,
   📊 results, ⏱️ time cost, 📚 data, 🌐 network, 📦 model/data wrap, 🖥️ system, 📌 info.
5. **Define the overrides in the order of the parent.** If a base class defines
   `prepare`, `forward`, then `sample`, every subclass defines them in that order. A
   helper that overrides nothing can be anywhere, but it is usually next to the method
   that uses it.
6. **Cite a paper equation as `see paper Eq. 10`.** For example,
   `loss = (v_pred - v_target).square().mean()  # see paper Eq. 10`. Here, "paper" is
   the paper that the file implements. For any other paper, use its name, for example
   `see GPT-3 Eq. 2`.
7. **Put the quotes of a multi-line docstring on their own lines.** A one-line
   docstring stays `"""Summary."""`. A multi-line one looks like this:
   ```python
   """
   Generate samples with an Euler solver from t = 0 to t = 1.

   One step uses t = 0 only. More steps move t from 0 to 1 in num_steps.
   """
   ```
