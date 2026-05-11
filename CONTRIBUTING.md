# Contributing

This repository is organized as research code. Contributions should preserve
reproducibility and avoid committing large generated artifacts.

Before opening a pull request:

1. Run `make check`.
2. Do not commit datasets, model weights, embeddings, checkpoints, or result bundles.
3. Document new experiment outputs in `docs/REPRODUCIBILITY.md` or a focused report.
4. Keep old experiment entry points backward compatible unless a breaking change is explicitly documented.

