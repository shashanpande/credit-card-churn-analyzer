# Credit Card Churn Analyzer

**Predicting customer churn in credit card portfolios using behavioral signals and machine learning.**

---

## The Problem

Credit card issuers lose significant revenue when customers churn — cancelling cards, going dormant, or shifting spend to competitors. Most churn signals exist in the data weeks or months before a customer leaves.

This project builds a simple, interpretable churn prediction system that:
- Identifies at-risk customers before they leave
- Surfaces the behavioral drivers of churn (not just who — but *why*)
- Segments the customer base into actionable risk tiers

---

## Key Findings

| Driver | Direction | Business Insight |
|---|---|---|
| Months Inactive (12M) | ↑ = higher churn | Inactivity is the #1 early warning signal |
| Transaction Volume (3M) | ↓ = higher churn | Declining frequency precedes cancellation |
| Avg Monthly Spend | ↓ = higher churn | Low spend = card isn't their primary |
| Reward Redemptions | ↓ = higher churn | Engagement with rewards = loyalty proxy |
| Number of Products | ↓ = higher churn | Multi-product customers churn less |

> **Churn is a behavior problem, not a price problem.** Customers don't leave because of fees — they drift away because they stop using the card.

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | ~0.97 |
| Recall (Churners) | ~0.93 |
| Precision (Churners) | ~0.88 |

Model: Random Forest Classifier with class balancing for 20% churn minority.

---

## Project Structure

```
credit-card-churn-analyzer/
│
├── credit_card_churn_analyzer.ipynb   # Main notebook (EDA + model + insights)
├── customer_churn_scores.csv          # Scored customers with risk tiers
├── churn_feature_importances.csv      # Feature importance rankings
│
├── outputs/
│   ├── churn_eda_distributions.png    # Behavioral distributions by churn label
│   ├── churn_confusion_matrix.png     # Model evaluation
│   ├── churn_feature_importance.png   # Top churn drivers chart
│   └── churn_risk_distribution.png    # Risk tier segmentation
│
└── README.md
```

---

## Notebook Structure

| Section | Content |
|---|---|
| 1. Setup | Imports, config, reproducibility |
| 2. Data Schema | Feature definitions and rationale |
| 3. Synthetic Data | Realistic customer data generation with behavioral differentiation |
| 4. EDA | Visual comparison of churned vs retained customers |
| 5. Preprocessing | Feature engineering and train/test split |
| 6. Model | Random Forest with class balancing |
| 7. Feature Importance | Key churn drivers with business interpretation |
| 8. Risk Scoring | Customer-level churn probability and risk tier assignment |
| 9. Insights | Business findings and retention action recommendations |
| 10. Export | CSV outputs for downstream use |

---

## Retention Action Framework

| Tier | Threshold | Recommended Action |
|---|---|---|
| High Risk | ≥ 0.70 | Immediate outreach — fee waiver, limit review, personalized offer |
| Medium Risk | 0.40 – 0.70 | Proactive engagement — rewards nudge, spend bonus, cross-sell |
| Low Risk | < 0.40 | Standard lifecycle marketing |

---

## Stack

- **Python 3.10+**
- `pandas` — data manipulation
- `numpy` — synthetic data generation
- `scikit-learn` — modeling and evaluation
- `matplotlib` / `seaborn` — visualization

---

## Setup

```bash
git clone https://github.com/yourusername/credit-card-churn-analyzer
cd credit-card-churn-analyzer
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook credit_card_churn_analyzer.ipynb
```

---

## Next Steps (v2)

- [ ] SHAP values for per-customer explainability
- [ ] Decision threshold optimization (precision/recall business tradeoff)
- [ ] Streamlit dashboard for stakeholder demos
- [ ] Cohort-level churn analysis
- [ ] Survival analysis — model *when*, not just *if*

---

## About

Built as part of an AI portfolio focused on applying machine learning to banking, credit, and customer intelligence problems.

**Domain:** Credit Cards · Customer Retention · Behavioral Analytics  
**Approach:** Simple MVPs with real business framing over complex architectures

---

*If you found this useful, connect with me on [LinkedIn](https://linkedin.com/in/yourprofile).*
