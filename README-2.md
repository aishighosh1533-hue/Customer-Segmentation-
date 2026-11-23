# Customer Segmentation, Coupon Response and Price Elasticity

A retail analytics project on the Dunnhumby *The Complete Journey* grocery data. It answers two questions a retailer faces every week:

1. **Which customers respond to coupons?** Segment households and predict who redeems a coupon.
2. **Which discounts are worth giving?** Estimate price elasticity by category and work out the margin each category needs for a discount to pay off.

**Tools:** Python, pandas, scikit-learn, XGBoost, statsmodels, matplotlib

---

## Project 1: Customer segmentation and coupon response

**Approach**
- Built one row per household from transactions made **before the first campaign (day 224)**, so no future information leaks into the features.
- Features: recency, frequency, monetary value (RFM), average basket size, own-brand share. A sixth feature, discount ratio, is used in the prediction models only.
- Log-transformed and standardised the features, then grouped households with **KMeans**. The number of segments (4) was chosen using elbow and silhouette plots.
- Predicted which targeted households redeem a coupon with **logistic regression, random forest and XGBoost**, compared by 5-fold stratified cross-validation (AUC).
- Evaluated targeting with a **gain curve**: if only the highest-scoring households are mailed, what share of redeemers is reached?

**Results**

| Segment | Households | Reading | Redemption rate |
|:--|:--|:--|:--|
| 1 | 800 | Big-basket loyal shoppers | 34.7% |
| 2 | 731 | Frequent small-basket shoppers | 24.9% |
| 3 | 613 | Occasional shoppers drifting away | 13.9% |
| 4 | 351 | Barely active | 9.3% |

The segment names are my reading of the averages, not something the data provides.

| Model | AUC (5-fold CV) |
|:--|:--|
| Logistic regression | 0.702 |
| XGBoost | 0.675 |
| Random forest | 0.666 |

Mailing the top 30% of households by score (474 of 1,583) reaches 221 of 433 redeemers (51%), a **1.7x lift** over random mailing.

## Project 2: Price elasticity and discount break-even

**Approach**
- Took the 12 largest grocery categories by sales and the 10 best-selling products in each (products sold in at least 60 weeks).
- Built a product-by-week panel of units and average price, with display and mailer promotion flags.
- Estimated price elasticity for each category with a log-log **fixed-effects regression** (product and week fixed effects, robust standard errors).
- With no cost data, calculated the **break-even margin**: the gross margin a category needs for a 5% price cut to raise profit.
- Compared break-even margins with assumed grocery margins of 20%, 30% and 40%, and labelled each category "Discount may pay", "Test first" or "Hold price".

**Results**
- Elasticity ranges from **-1.2** (bread, bag snacks) to **-3.4** (fluid milk).
- Break-even margin ranges from **31%** (fluid milk) to **83%** (bread).
- Only fluid milk and soft drinks come close to typical grocery margins, so discounts in the other categories should be held back.

## Data

Dunnhumby *The Complete Journey* (available on Kaggle). The data is **not included** in this repository. Download it and place these files in a `data/` folder:

- `transaction_data.csv`
- `product.csv`
- `campaign_desc.csv`
- `campaign_table.csv`
- `coupon_redempt.csv`
- `causal_small.csv.gz` (product-week display and mailer flags, used for Project 2)

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn xgboost statsmodels jupyter
jupyter notebook
```

Open the notebook, set `DATA_PATH = "data/"` in the data-loading cells, and run the cells from the top. The notebook reads about 2.6 million transaction rows, so loading takes a minute or two.

## Repository structure

```
.
├── README.md
├── Customer_Segmentation_Coupon_Response_Price_Elasticity.ipynb
└── data/            (not tracked: download from Kaggle)
```

## Limitations

- **Small coupon sample.** Models were trained on 1,583 targeted households (433 redeemers), so the AUC is modest and varies between folds (about 0.70 plus or minus 0.05). Logistic regression beating the more complex models fits this.
- **Redemption is not the same as response caused by the coupon.** The models rank who tends to redeem. They do not show that the coupon changed behaviour.
- **Elasticities show association.** Prices and promotions change together in retail data, so some of the price effect may be promotion effect. The fluid milk estimate is unusually high for a staple, so treat the figures as a comparison across categories more than as exact values.
- **Uneven precision.** Beers and ales has a very wide interval (-2.87 to -0.19), so its break-even margin is unreliable.
- **Margins are assumed.** The data has no cost information. The 20%, 30% and 40% margins are assumptions, not data.
- **Limited product coverage.** Elasticities use the 10 best-selling products in each category, not every product.

## Author

Aishi Ghosh, MA General Economics, Madras School of Economics.
