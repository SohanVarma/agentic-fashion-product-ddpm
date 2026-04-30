""# Agentic Conditional Fashion Product Image Generator using DDPM

This project is a Generative AI system for fashion product image generation using a class-conditional DDPM with an agentic workflow.

## Milestone 1 (April 30, 2026)

- Domain note → `domain_note.pdf`
- Data pipeline → `src/milestone1_data_pipeline.py`
- Initial model → `src/milestone1_smoke_test.py`
- Results → `results/`

Run:

```bash
python src/milestone1_data_pipeline.py
python src/milestone1_smoke_test.py
```

This satisfies Milestone 1 requirements: data loaded, pipeline working, and model producing logged results.

## Dataset

Kaggle Fashion Product Images Small

```bash
python download_dataset.py
```

## Full system

Includes DDPM training, classifier, and agent pipeline (intent → planner → generation → evaluation → improvement → report).
""