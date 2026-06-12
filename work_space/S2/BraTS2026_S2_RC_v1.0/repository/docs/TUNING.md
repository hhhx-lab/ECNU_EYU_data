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

The validation split should remain fixed.

Do NOT modify the train/validation split during tuning.

