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

Use the preserved fold 0 validation set for hyperparameter selection.

Do not tune independently on folds 1-4. After selecting one configuration on
fold 0, freeze it and train folds 1-4 for cross-validation and out-of-fold
evaluation. Do not modify any fold files during tuning.
