# Executive Summary: Predicting Bank Customer Churn

**Audience:** policy makers, regulators and senior bank managers (no technical background needed)

## The question
Banks lose revenue and long-term relationships when customers leave ("churn"). Can data help a bank spot, *before* they leave, the customers most at risk, so that retention effort can be targeted instead of broad?

## What was done
A machine-learning prototype was built on 10,000 customer records from France, Germany and Spain. About 1 in 5 customers (20.37%) had left. Four prediction methods were tested on 2,000 customers the models had not seen. The best-performing method for catching leavers was also turned into an interactive dashboard where an analyst can enter a customer profile, see a risk score between 0% and 100%, and test "what if" changes.

## Key findings
- **The models work reasonably well, but not perfectly.** The strongest method ranks at-risk customers correctly about 87% of the time (ROC-AUC 0.867). The simplest method found fewer than 1 in 5 leavers, so method choice matters.
- **A trade-off is built in.** The dashboard model (Random Forest) found 243 of 407 customers who left and missed 164. It also flagged 154 customers who stayed. Flagging more customers catches more leavers but wastes more outreach; the right balance depends on what retention efforts cost.
- **What the data associates with churn** (patterns in this dataset, not proven causes):
  - Customers in Germany left more often (32.4%) than in France or Spain (about 16%).
  - Inactive members left more often (26.9%) than active members (14.3%).
  - Older customers left more often; risk rises sharply from the early 40s.
  - Customers with 3 or 4 products left very often, but these groups are small (266 and 60 customers) and need checking before any action.

## Recommendations
1. Use the scores to help analysts decide **who to contact first**, not to make automatic decisions about customers.
2. Set the risk threshold using the real cost of a missed leaver versus an unnecessary offer.
3. Check results by customer group (for example country and gender) for fairness before wider use.
4. Test the model on newer data, and confirm the data's source and sharing rights.
5. Investigate why customers in Germany and with 3 or 4 products leave more.

## Limits
This is an educational prototype tested on one dataset. Its explanations show how the model behaves, not why customers actually leave. It should not be used for real decisions without further validation.
