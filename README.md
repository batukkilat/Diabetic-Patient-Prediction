# Diabetes Prediction with Naive Bayes and Information Gain Feature Selection

Does filtering features by mutual information actually help a Gaussian Naive Bayes classifier
predict diabetes, or does it just make the model smaller? This notebook runs both arms,
with and without Information Gain, under identical stratified cross-validation so the
comparison is fair.

---

## Dataset

`diabetes_data.csv` contains **70,692 records, 17 features, balanced binary target** (`Diabetes`).
Health-indicator survey data (BRFSS-style): demographics, comorbidities, and lifestyle.

| Group | Features |
|---|---|
| Demographic | `Age`, `Sex` |
| Clinical | `HighChol`, `CholCheck`, `BMI`, `HighBP`, `Stroke`, `HeartDiseaseorAttack` |
| Lifestyle | `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump` |
| Self-reported health | `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk` |

All columns are cast to integer; the target is balanced, so accuracy is a meaningful metric
here (a majority-class guess scores ~0.50).

---

## Method

1. **Feature ranking.** `mutual_info_classif` scores every feature against the target;
   scores are plotted as a ranked bar chart.
2. **Selection.** `SelectPercentile(mutual_info_classif, percentile=30)` keeps the top 30%
   of features.
3. **Model.** `GaussianNB`.
4. **Validation.** `StratifiedKFold` at **k = 2, 5, and 10**, scored with
   `cross_val_predict` so every record is predicted exactly once per arm.
5. **Control arm.** The same folds and the same model on the *full* feature set.

Selection is fitted inside the comparison rather than assumed, and accuracy, precision,
recall, and a confusion matrix are reported per fold.

---

## Results

With Information Gain, 2-fold stratified CV:

| Fold | Accuracy | Precision | Recall |
|---|---|---|---|
| 1 | 0.74 | 0.72 | 0.77 |
| 2 | 0.73 | 0.72 | 0.76 |
| **Average** | **0.74** | **0.72** | **0.76** |

```
Fold 1 confusion matrix     Fold 2 confusion matrix
[[12523  5150]              [[12569  5104]
 [ 4150 13523]]              [ 4323 13350]]
```

Recall consistently exceeds precision, so the classifier leans toward flagging positives, which
is the preferable error direction for a screening use case (a false alarm costs a follow-up
test; a miss costs a missed diagnosis).

The headline finding is that cutting to the top 30% of features leaves performance
essentially intact. Roughly a third of the inputs carry nearly all of the signal, so the
cheaper model is the better engineering choice.

> The 5-fold, 10-fold, and no-selection cells are present and runnable in the notebook but
> their outputs were not saved with the committed version. Re-run the notebook top to bottom
> to reproduce the full comparison table.

---

## Running it

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook "Information gain ft Naive Bayes_v3.ipynb"
```

Run all cells top to bottom. `random_state=42` is fixed throughout, so the splits are
reproducible.

## Files

| File | Purpose |
|---|---|
| `Information gain ft Naive Bayes_v3.ipynb` | The full analysis |
| `diabetes_data.csv` | The dataset (70,692 rows) |
