# Telco Churn: Who to Contact, With What Action, and What It Is Worth

Most churn projects stop at "the model has an AUC of 0.85". This one starts from the decision a retention team actually has to make each month:

> *Which customers should we contact this month, with what action, and what is that worth?*

It uses the public [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 columns). The analysis produces a ranked contact list, a business-value estimate for every list size, and concrete actions per customer segment.

## Key results

All results are on a held-out test set of 1,409 customers that was used once, at the end.

| | |
|---|---|
| **Top 20% by predicted risk** | 68% are real leavers (random: 27%); the list reaches 52% of all churners |
| **Model vs a simple business rule** | at the same 291 contacts: 194 vs 170 churners (+14%) for the rule "month-to-month & tenure ≤ 6" |
| **Best contact list** (ranked by risk × value) | top 23% → ≈ €4,750 per 1,000 customers per monthly campaign, ≈ 4× the rule's value |
| **Model choice** (nested CV) | logistic regression vs XGBoost: PR-AUC 0.673 vs 0.668, ROC-AUC 0.850 vs 0.849. They are tied, so the explainable model was chosen |
| **Generalisation** | test PR-AUC 0.665 / ROC-AUC 0.847 vs 0.674 / 0.850 in CV |

**What drives churn** (logistic regression, odds ratios with 95% CIs):
- **Contract and tenure both matter.** New month-to-month customers churn at 54%. At the same tenure, the average predicted risk is 31% on month-to-month vs 10% on a two-year contract.
- **Risk is highest early:** 50% in month 1 → 28% at 12 months → 16% at 4 years.
- **Fiber churns more than DSL** (38% vs 23% what-if risk), but price and product can't be separated in this data.
- Electronic-check payment and missing tech support / online security also raise risk. **Gender has no effect** (OR 1.02, CI 0.89–1.18).

## Approach

**1. Frame the problem as a ranking, judged in €.** Predicting "nobody leaves" is already 73.5% accurate, so accuracy and the 0.5 threshold are not used. Customers are ranked by predicted risk × monthly charge. The contact budget decides how far down the list to go.

**2. Leakage-safe workflow.**
- The data is split 80/20 (stratified) *before* EDA.
- Anything learned from data (scaling, encoding) sits inside scikit-learn pipelines fitted on training folds only.
- The test set is used once.

**3. Hypotheses before modelling.** Eight hypotheses (H1–H8) were written down from the EDA and tested on the final model. The comparison rule for H8 was committed before any modelling.

**4. Model comparison with nested CV.** Five outer folds score the models; five inner folds tune them.

| Model | Role |
|---|---|
| Dummy (prior) | sanity check |
| Rule: month-to-month & tenure ≤ 6 | business floor |
| Logistic regression (tenure + log tenure) | **final model**, interpretable |
| XGBoost (light grid) | flexible challenger |

PR-AUC selects the model because it rewards the top of the list. ROC-AUC, log-loss and "% of churners reached in the top 20%" are also reported.

**5. Design checks**, each with the evidence in the notebook:

| Check | Decision |
|---|---|
| VIF: `MonthlyCharges` is the sum of the services (R² = 0.999, VIF ≈ 860) | left out of the logistic regression; kept for customer value |
| Tenure shape (linear / log / bands / both) | tenure + log tenure: higher PR-AUC in all 5 folds, captures the month-1 spike |
| `SeniorCitizen`, `gender` | not model inputs (no added value / protected); used for the fairness audit |
| Class reweighting (`class_weight="balanced"`) | not used: ranking unchanged (98% same top 20%), but probabilities inflated from 27% to 41% |
| Regularisation C, robustness seeds | flat CV curve; results move by ≤ 0.005 across five pre-fixed seeds |
| Calibration | predicted ≈ observed on out-of-fold and test data, so the € calculation can use the probabilities |

**6. Business translation.**
- **Expected value** of a contact = P(churn) × offer success rate × 12 months × margin × monthly charge − contact cost.
- **Base assumptions:** 20% success rate, 40% margin, €30 per call. These are not in the data, and every one is varied in a sensitivity table.
- **Profit curve:** the value of every possible list size, compared with random selection and the rule.
- **Tiers** follow the profit curve: **Call** = top 23%, **CRM email/SMS** = up to 40%, **No outreach** = the rest.

| Tier (test back-test) | Customers | Actual churn | Profile |
|---|---|---|---|
| Call | 324 | 61% | 98% month-to-month, 99% fiber, 69% e-check, 43% in first 6 months |
| CRM | 239 | 35% | |
| No outreach | 846 | 11% | |

**Recommendation:**
- Start an early-life programme for new month-to-month fiber customers: a check-in call plus a contract upgrade offer.
- Validate it with a **4–6 week A/B pilot**. The measured success rate then sets the list size on the profit curve.

## Limitations
- **One snapshot without dates.** The score measures resemblance to past leavers, not a time-based forecast.
- **The € values depend on the offer success rate.** At 10% success only the top ~2% pays off; at 40%, contacting ~40% does.
- **Fairness audit:** gender is balanced in the contact list (50/50). Senior citizens are over-represented (27% of the list vs 16% of customers) because their risk is higher; their hit rate in the list is 56% vs 48%, so they are not mistargeted, but the share should be monitored.
- **High risk is not the same as persuadable.** Logging offers and responses would allow an uplift model later.

## Repository structure

```
telco_churn_retention.ipynb   full analysis, sections 1–9 with explanations and outputs
data/                         place the Kaggle CSV here (not included)
outputs/figures/              EDA, gains, calibration and profit-curve charts
outputs/results/              contact list, test IDs, gains and profit curves (CSV)
requirements.txt
```

## Running it

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) into `data/`.
2. Install the requirements and run the notebook:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook telco_churn_retention.ipynb
```

Python 3.13. A full run takes a few minutes, most of it in the nested cross-validation. All randomness is seeded (`RANDOM_STATE = 42`).

## Tech stack

Python · pandas · scikit-learn · XGBoost · statsmodels · matplotlib · seaborn
