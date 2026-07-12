# S2 Fixed Splits

Cross-validation is disabled.

```text
data/splits/current/  Dataset263 current patient-group split, 823/103/104
data/splits/legacy/   Dataset260 historical recovery, 828/207/259
```

The current split is authoritative for new real-only and real+synth paired experiments. The legacy split exists only to audit the completed historical checkpoint. Both modes run raw/preprocessed ID-space validation before training.
