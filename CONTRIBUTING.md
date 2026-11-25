# Contributing to FakeShield AI

Thanks for your interest. The project is a research prototype, so ideas, bug reports and feedback on the experimental protocol are as valuable as code.

## Ground rules

- **Measure before you claim.** A change that is said to improve results must come with the metrics (accuracy, macro-F1, ROC-AUC) and the seed it was measured with.
- **Never present preliminary numbers as final.** Mark them as preliminary until they are reproduced.
- **Keep the split fixed.** The stratified 70/15/15 split uses `split_seed = 42`; do not tune on the test set.

## Workflow

```bash
git clone https://github.com/Chaymae-El-Bahloul/FakeShield-AI.git
cd FakeShield-AI
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

1. Open an issue describing the problem or the idea.
2. Create a branch: `git checkout -b feat/short-description`.
3. Make small, focused commits using [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
4. Make sure `python -m pytest -q` passes, then open a pull request.

## Good first contributions

- Evaluate on NewsCLIPpings (cross-dataset)
- Add multilingual support (Arabic, Moroccan Darija)
- Calibration and uncertainty estimation
- More tests for the data loader and the training loop
