---
paths:
  - "README.md"
---

# README

Follow these conventions in the top-level `README.md`. They add to
[common-readme](common-readme.md).

1. **Sections:** put the Algorithms section after Updates. A newly reproduced paper
   gets an Updates line.
2. **Algorithms:** one HTML table of the reproduced papers, with the columns Algorithm,
   Paper, Released, and Ours / Paper. Keep the table in `reproduce/README.md` the same.
   Write the HTML as [common-markdown](common-markdown.md) says.
   - **Groups:** one bold cell across all columns starts each group of a domain and a
     category:
     `<td colspan="4" align="center"><b>Visual Generation - Paradigms</b></td>`.
   - **Order:** in each domain, put the Paradigms group first and the Post-Training
     group last. Order the papers of each group by the Released date, newest first.
   - **Algorithm and Paper:** the Algorithm cell links the name to the paper's README
     under `reproduce/`. The Paper cell cites the paper as
     [common-markdown](common-markdown.md) says, in HTML:
     ```html
     <td nowrap><a href="meanflow">MeanFlow</a></td>
     <td>
       <i>Mean Flows for One-step Generative Modeling</i>
       (<a href="https://arxiv.org/abs/2505.13447">NeurIPS 2025</a>)
     </td>
     ```
   - **Released:** the date when the paper first became public, as `YYYY-MM-DD`. This
     is its first arXiv version, or its venue if that is earlier. It is not the date
     of our reproduction.
   - **Ours / Paper:** one to three main results, one per line with `<br>`, for
     example `B/2 FID: 6.26 / 6.17` or `GenEval: 0.95 / 0.98`. Use the main metric of
     the paper, and the rows of the paper table in its order. Prefer our runs at the
     paper setting. If the table has more rows, keep the largest models or the first
     tasks. Copy both numbers from the same column of the Results table of the paper
     README. Write `-` for a value that the paper does not report.
   - **One line:** give the Algorithm, Released, and Ours / Paper cells the `nowrap`
     attribute.
   - **New paper:** add its row after we record its first run.
3. **Quick Start:** one short train, evaluate, and kill example (MeanFlow-B/2).
