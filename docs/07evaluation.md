# Evaluation Guide

---

## 1. Evaluate a Trained Model

Use the MLflow run ID:

```bash
python scripts/evaluate.py logging.run_id=<run_id>
```

Outputs:

- Dice 
- IoU 
- Accuracy 
- Sensitivity 
- Specificity