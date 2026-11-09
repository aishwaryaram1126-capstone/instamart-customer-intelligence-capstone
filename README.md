# Instamart Customer Intelligence Capstone

## Overview

This project builds a customer intelligence and support automation solution using synthetic Instamart data.

The project covers:
- Data cleaning and database creation.
- Customer segmentation and lapse prediction.
- Support ticket classification with privacy guardrails.
- Policy-based refund assistance using a ReAct agent.

All generated data is synthetic and fictional. No real customer information, external API key, or real network call is used.

## Reproduce

Run the following commands from the project root directory.

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment on Windows:

```powershell
.venv\Scripts\activate
```

3. Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

4. Generate the synthetic data:

```bash
python generate_data.py
```

5. Run the Part A data preparation script:

```bash
python src/part_a.py
```

6. Run the Part B customer intelligence script:

```bash
python src/part_b.py
```

7. Run the Part C ticket triage script:

```bash
python src/part_c.py
```

8. Run the Part D refund agent:

```bash
python src/part_d.py
```

9. Run the final validation script:

```bash
python src/validate.py
```

**Note:** The Part A–D scripts and validation script will be created during the project. We will update these commands if the final script names differ.


## Part A — Data Foundation: Clean, Store, Query

### Data cleaning
- Generated and cleaned synthetic Instamart customer, product and order data.
- Clean orders: 12,309.
- Completed orders: 11,110.
- Cancelled orders: 1,199.
- Completed order revenue: INR 2,998,668.80.

### SQLite database
Database: `data/instamart.db`

Tables:
- customers
- products
- orders

### SQL business analysis
SQL queries are available in `sql/queries.sql`.
Database validation and seven SQL business queries were completed.

## Part B — Customer Intelligence

### RFM analysis
- RFM analysis was performed using customer purchase behaviour.
- Customer segmentation was completed using K-Means clustering.
- Selected number of clusters: 4.
- PCA explained variance: 85.23%.

### Lapse prediction
- Lapse cohort: 1,156 customers.
- Lapsed customers: 443.
- Lapse rate: 38.32%.

### Model evaluation
- Random Forest cross-validation F1 score: 0.68105.
- Oversampled Logistic Regression test F1 score: 0.70.
- Random Forest test F1 score: 0.655.

Generated outputs include:
- `outputs/customer_segments.csv`
- `outputs/retention_list.csv`
- `outputs/lapse_model.joblib`

## Part C — Support Ticket Triage with Guardrails

### Input guardrails
- Processed 164 synthetic support tickets.
- 156 tickets were ready for classification.
- 4 tickets were empty.
- 4 tickets exceeded the input length limit.

### Privacy redaction
PII redaction results:
- Emails: 55
- Phone numbers: 91
- Order IDs: 132
- Total redactions: 278

### Classification and validation
- Zero-shot classification accuracy: 85.71%.
- Few-shot classification accuracy: 100%.
- Tickets were routed into support queues based on their predicted categories.

Final routing:
- Support queue: 102
- Security team: 34
- Human review: 17
- Premium desk: 11

### Token budget and retry handling
Token budget demonstration was completed using `outputs/triage_budget_demo.csv`.

Classification results are saved in `outputs/triage_results.csv`.

## Part D — Policy RAG and Refund ReAct Agent

### Policy document retrieval
- Loaded 5 policy documents.
- Created 15 text chunks.
- Chunk size: 40 words.
- Chunk overlap: 10 words.
- Implemented TF-IDF vectorization and cosine similarity.
- Retrieved the top 3 policy chunks for refund-related queries.

### RAG evaluation
With 10-word overlap:
- Document retrieval: 12/12.
- Expected fact found: 12/12.
- Correct fallback: 2/2.

With zero overlap:
- Document retrieval: 12/12.
- Expected fact found: 10/12.
- Correct fallback: 2/2.

### Refund agent tools
Implemented five tools:
- `get_order`
- `search_policy`
- `calculate_refund`
- `issue_refund`
- `escalate_to_human`

Refund actions are recorded in `outputs/refund_ledger.jsonl`.

### Scenario evaluation
Seven scenarios were executed:

| Scenario | Result |
|---|---|
| S1 | REFUND — INR 128.25 |
| S2 | REFUND — INR 135.00 |
| S3 | ESCALATE |
| S4 | DENIED |
| S5 | DENIED |
| S6 | NOT_FOUND |
| S7 | FALLBACK_MAX_ITERATIONS |

### Stop-safely demonstrations
Both safety limit tests passed:
- Token limit: `FALLBACK_TOKEN_LIMIT`
- Timeout: `FALLBACK_TIMEOUT`

### Crash and resume demonstration
- Initial status: `PAUSED_FOR_RESUME`
- Tools completed before pause: 2
- Resumed status: `REFUND`
- Final answer: INR 128.25
- Total tools completed: 4

Checkpoint and resume test passed.

### ReAct versus plan-and-execute
The implemented agent uses a ReAct-style approach. The model requests tools and receives observations iteratively.

A separate measured plan-and-execute benchmark has not yet been completed. Therefore, no comparative model-call or token-efficiency figures are claimed.

## Limitations and Next Steps

- The dataset is synthetic and may not represent real customer behaviour.
- Model results depend on the generated dataset and selected parameters.
- The retention list is illustrative and is not a leakage-free evaluation because the Premium flag uses whole-window RFM information.
- Future improvements could include testing on real, appropriately anonymised data and monitoring model performance over time.

## Conclusion

The Instamart Customer Intelligence Capstone demonstrates an end-to-end workflow covering data preparation, customer analytics, machine learning, privacy-aware ticket classification, policy retrieval and refund automation.

All four project parts have been implemented and their reported tests have been executed.
