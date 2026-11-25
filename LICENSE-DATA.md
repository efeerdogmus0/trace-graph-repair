# Licence for data, figures, text and adapter weights

Copyright (c) 2026 Yiğit Efe Erdoğmuş

The following are licensed under the Creative Commons Attribution 4.0 International licence
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/):

- the generated benchmark data: graph records, questions, answers, rendered images and their
  manifests (everything the generator in `src/trace_vr` produces, and the copies in `data/` and `manifests/`);
- reports, tables and figures (`reports/`, `paper/figures/`), the paper text (`paper/`) and the documentation;
- the LoRA adapter weights trained in this project, when released.

You may share and adapt them for any purpose, including commercially, if you give appropriate
credit (cite the paper or link this repository), link the licence and say whether you made changes.

Source code (`src/`, `scripts/`, `tests/`, `notebooks/`, `demo/`) is under the MIT licence in `LICENSE`.

Not covered by either licence:

- the base model Qwen/Qwen3.5-2B: its own licence (Apache-2.0 per its model card) applies to the base
  weights, which are not redistributed here; an adapter is used together with them;
- the DejaVu fonts in `assets/fonts/`: their own licence, `assets/fonts/LICENSE_DEJAVU`;
- VGCure / MCDGraph: no file of that dataset or code is in this repository (ids, counts and hashes only);
  obtain it from its authors under their terms.

`THIRD_PARTY.md` lists every third-party component and the licence it declares.
