# Fake Review Detection in Online Restaurant Reviews Using Machine Learning and Reviewer Behaviour Analytics

Fraud Detection Analytics (FDA) project. A machine-learning model scores Yelp restaurant reviews for how suspicious they are, so human moderators can check the riskiest ones first. Uses **real** (not synthetic) data and **classic ML on structured data only** (no text/NLP).

## Results (held-out test set, 2,483 reviews)

| Metric | Value |
|---|---|
| Accuracy | 0.865 |
| Precision (fake) | 0.850 |
| Recall (fake) | 0.894 |
| F1 | 0.871 |
| ROC-AUC | 0.934 |
| PR-AUC | 0.921 |

Final model: tuned **Gradient Boosting** (`HistGradientBoostingClassifier`) on 18 review-level and reviewer-profile features.
Numbers can differ in the third decimal with different library versions.

![Final evaluation](images/04_final_evaluation.png)

## Models compared (reviewer-grouped 5-fold CV + test set)

| Model | CV AUC | Test accuracy | Test ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.934 | 0.863 | 0.932 |
| Decision Tree | 0.924 | 0.851 | 0.923 |
| Random Forest | 0.936 | 0.874 | 0.937 |
| Gradient Boosting | 0.936 | 0.868 | 0.936 |

## Key findings

1. **Who wrote a review matters most.** Reviewer-profile features alone give 0.935 AUC; review-level features alone give only 0.69.
2. **A leaking column was found and excluded.** `reviewUsefulCount` is 0 for every flagged review (Yelp hides filtered reviews, so they get no votes) and only grows after posting. Including it would add a fake ~0.014 AUC.
3. **Features matter more than the algorithm.** Logistic Regression, Random Forest and Gradient Boosting are within noise of each other.
4. **Blind spot:** the model catches ~99.6% of fakes from new accounts (1-2 reviews) but only ~56% from reviewers with more than 10 reviews.
5. **Best use is ranking for human moderators.** On this ~50% fake sample precision is 0.85; at a realistic 13% fake rate it would be ~0.45.

![What detects fakes](images/03_ablation.png)
![Feature importance](images/06_importance.png)

## Method

| Step | What was done |
|---|---|
| Data checks | missing values, duplicates, train/test overlap, class balance, label-leak scan |
| EDA | reviewer activity, rating pattern, correlated features |
| Features | review-level (rating gap, extreme rating, length) and reviewer-level (activity, social ties, account age, reviews per month) |
| Validation | 5-fold CV **grouped by reviewer** (553 reviewers appear in both train and test) |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting; small grid search on Gradient Boosting |
| Evaluation | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, threshold table, segment analysis |
| Explainability | permutation importance and partial dependence |
| Extras | error analysis, precision at realistic fake rates, "top-k% for moderators" table |

## Limitations

* Labels are **Yelp's filter decisions**, not verified fraud.
* Data is ~50% fake; real-world precision would be much lower.
* About 100 restaurants, mostly reviewed by Chicago-area users.

## Files

| File | What it is |
|---|---|
| `fake_review_detection.ipynb` | the full project (data checks to conclusion), already executed |
| `fake_review_detection.py` | the same code as a plain Python script (saves figures to `images/`) |
| `data/yelp_train.csv`, `data/yelp_test.csv` | train (9,929) and test (2,483) reviews, comma-separated UTF-8 |
| `images/` | figures shown in this README |
| `requirements.txt` | Python packages |

**Data source:** Kaggle - *<add dataset name and link here>*. Original data from Yelp academic research releases; respect the original licence/terms.

## Run it

```bash
pip install -r requirements.txt

python fake_review_detection.py            # option 1: plain script, about 1-2 minutes
jupyter notebook fake_review_detection.ipynb   # option 2: notebook, then Run All
```

Or upload the notebook to Google Colab, create a `data/` folder there, upload the two CSVs into it, and choose Runtime > Run all.
