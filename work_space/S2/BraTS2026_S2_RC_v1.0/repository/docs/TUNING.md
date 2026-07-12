# Hyperparameter Tuning Notes

Current baseline:

RC CE weight = 3

Epochs = 1000

---

## Suggested future tuning

RC weight:

3
5
10

---

## Evaluation strategy

Train full model first.

After convergence:

compare RC sensitivity and Dice.

Select the best checkpoint.

---

## Important

Use the current fixed 103-case validation set for hyperparameter selection. Cross-validation is disabled. After selecting one configuration, freeze it and use the current 104-case internal locked test only for final internal review. Do not modify `data/splits/current/` during tuning. Historical Dataset260 metrics are reference-only and must not be mixed into current paired comparisons.
