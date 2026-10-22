---
paths:
  - "**/*"
---

# Code Style

- Write all code and comments in clear, concise, standard English.
- Line length is 88 (`ruff`).
- Naming: `-` joins fields, `_` is used inside a field. Registry keys and config `name` values
  are snake_case (`imagenet_1k`, `meanflowdit_b_2`); timestamps are `%Y%m%d_%H%M%S`.
  Example run dir: `imagenet_1k-meanflowdit_b_2-meanflow-20260903_180000`.
- Start every print message with an emoji that matches its meaning, following existing usage:
  🚀 running, ✅ success, ⚠️ warning, ⏳ in progress, 💾 saved, 🛑 exit, ❌ error, 📊 results,
  ⏱️ time cost, 📚 data, 🌐 network, 📦 wrap, 🖥️ system, 📌 info.
