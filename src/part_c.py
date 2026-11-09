import sys
import re
import json
import time
import random
from pathlib import Path

import pandas as pd

# Find the main project folder
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

# File locations
TICKETS_FILE = ROOT / "data" / "raw" / "tickets.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load tickets in their original order
tickets = pd.read_csv(TICKETS_FILE, keep_default_na=False)

print("PART C - SUPPORT TICKET TRIAGE")
print("--------------------------------")
print("Total tickets:", len(tickets))
print("\nColumns:")
print(tickets.columns.tolist())

print("\nFirst 5 tickets:")
print(tickets.head())

print("\nTicket data loaded successfully!")
# -----------------------------------------
# STEP 2: INPUT GUARDRAILS AND PII REDACTION
# -----------------------------------------

MAX_INPUT_TOKENS = 400

# Exact redaction patterns, applied in the required order
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


def redact_pii(text):
    """Redact email, phone number and order ID."""

    counts = {
        "emails": 0,
        "phones": 0,
        "order_ids": 0
    }

    text, counts["emails"] = re.subn(
        EMAIL_PATTERN, "[EMAIL]", text
    )

    text, counts["phones"] = re.subn(
        PHONE_PATTERN, "[PHONE]", text
    )

    text, counts["order_ids"] = re.subn(
        ORDER_PATTERN, "[ORDER_ID]", text
    )

    return text, counts


# Store redacted tickets and count substitutions
redaction_totals = {
    "emails": 0,
    "phones": 0,
    "order_ids": 0
}

processed_tickets = []

for _, ticket in tickets.iterrows():

    original_text = str(ticket["text"])

    # Check empty tickets
    if not original_text.strip():
        status = "skipped_empty"
        redacted_text = ""
        counts = {
            "emails": 0,
            "phones": 0,
            "order_ids": 0
        }

    else:
        # Redact PII
        redacted_text, counts = redact_pii(original_text)

        # Check for remaining PII
        assert not re.search(EMAIL_PATTERN, redacted_text)
        assert not re.search(PHONE_PATTERN, redacted_text)
        assert not re.search(ORDER_PATTERN, redacted_text)

        # Count substitutions
        for key in redaction_totals:
            redaction_totals[key] += counts[key]

        # Check input token limit
        if mock_llm.est_tokens(redacted_text) > MAX_INPUT_TOKENS:
            status = "flagged_long"
        else:
            status = "ready"

    processed_tickets.append({
        "ticket_id": ticket["ticket_id"],
        "customer_id": ticket["customer_id"],
        "order_id": ticket["order_id"],
        "true_category": ticket["true_category"],
        "original_text": original_text,
        "redacted_text": redacted_text,
        "status": status
    })


processed_df = pd.DataFrame(processed_tickets)

print("\nINPUT GUARDRAIL RESULTS")
print("----------------------")
print(processed_df["status"].value_counts())

print("\nPII REDACTION TOTALS")
print("-------------------")
print(redaction_totals)

print(
    "Total PII substitutions:",
    sum(redaction_totals.values())
)

print("\nGuardrails and redaction completed!")
# -----------------------------------------
# STEP 3: ZERO-SHOT AND FEW-SHOT PROMPTS
# -----------------------------------------

CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT_PROMPT = """
You are a customer support ticket CLASSIFY assistant.

Classify the customer's message into exactly one category:
billing, delivery, technical, fraud.

Return JSON only, using this exact format:
{"category": "...", "reason": "..."}

Do not include markdown or additional text.
"""

FEW_SHOT_PROMPT = """
You are a customer support ticket CLASSIFY assistant.

Classify the customer's message into exactly one category:
billing, delivery, technical, fraud.

Return JSON only, using this exact format:
{"category": "...", "reason": "..."}

Do not include markdown or additional text.

EXAMPLES:
Ticket: I was charged twice for my order.
Answer: {"category": "billing", "reason": "Duplicate charge"}

Ticket: My account was hacked and someone placed an order.
Answer: {"category": "fraud", "reason": "Unauthorized account activity"}

Ticket: The delivery rider has not arrived.
Answer: {"category": "delivery", "reason": "Delivery delay"}
"""

# Check prompt requirements
assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

for category in CATEGORIES:
    assert category in ZERO_SHOT_PROMPT
    assert category in FEW_SHOT_PROMPT

assert '"category"' in ZERO_SHOT_PROMPT
assert '"reason"' in ZERO_SHOT_PROMPT

print("\nCLASSIFICATION PROMPTS CREATED")
print("--------------------------------")
print("Zero-shot prompt: PASSED")
print("Few-shot prompt: PASSED")
print("All four categories included: PASSED")
print("JSON output format: PASSED")
# -----------------------------------------
# STEP 4: RETRY SYSTEM AND OUTPUT VALIDATION
# -----------------------------------------

BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_model_with_retry(messages, ticket_id):
    """Call mock LLM with up to 3 retries."""

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            return response

        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                raise

            wait_time = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Ticket {ticket_id}: "
                f"retry {attempt + 1}/{MAX_RETRIES}, "
                f"waiting {wait_time:.3f} seconds"
            )

            time.sleep(wait_time)


def validate_model_output(response_text):
    """Validate JSON first, then try keyword fallback."""

    try:
        result = json.loads(response_text)

        if isinstance(result, dict):
            category = result.get("category", "")

            if category in CATEGORIES:
                return (
                    category,
                    "classified_json",
                    result.get("reason", "")
                )

            # Valid JSON but invalid category
            return ("", "flagged_invalid_output", "")

        return ("", "flagged_invalid_output", "")

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", response_text.lower())

        found_categories = [
            word for word in words if word in CATEGORIES
        ]

        if len(found_categories) == 1:
            return (
                found_categories[0],
                "classified_keyword",
                "Recovered from keyword fallback"
            )

        return ("", "flagged_invalid_output", "")


def classify_ticket(ticket, system_prompt):
    """Classify one redacted ticket."""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": ticket["redacted_text"]
        }
    ]

    # Verify the user message contains only redacted text
    assert messages[-1]["content"] == ticket["redacted_text"]

    response = call_model_with_retry(
        messages,
        ticket["ticket_id"]
    )

    response_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens

    category, status, reason = validate_model_output(
        response_text
    )

    return {
        "category": category,
        "status": status,
        "reason": reason,
        "tokens_used": tokens_used
    }


print("\nRETRY AND VALIDATION FUNCTIONS CREATED")
print("--------------------------------------")
print("Maximum retries:", MAX_RETRIES)
print("Base delay:", BASE_DELAY)
print("JSON validation: READY")
print("Keyword fallback: READY")
print("Invalid output handling: READY")
# -----------------------------------------
# STEP 5: RUN ZERO-SHOT AND FEW-SHOT
# -----------------------------------------

def run_classification(system_prompt, run_name):
    """Run classification on all tickets in file order."""

    # Reset mock LLM immediately before each full run
    mock_llm.reset()

    results = []

    for _, ticket in processed_df.iterrows():

        # Empty and long tickets must never reach the model
        if ticket["status"] != "ready":
            results.append({
                "ticket_id": ticket["ticket_id"],
                "customer_id": ticket["customer_id"],
                "true_category": ticket["true_category"],
                "category": "",
                "status": ticket["status"],
                "tokens_used": 0
            })
            continue

        result = classify_ticket(ticket, system_prompt)

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "category": result["category"],
            "status": result["status"],
            "tokens_used": result["tokens_used"]
        })

    results_df = pd.DataFrame(results)

    classified = results_df[
        results_df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    correct = (
        classified["category"] == classified["true_category"]
    ).sum()

    accuracy = (
        correct / len(classified) if len(classified) else 0
    )

    total_tokens = results_df["tokens_used"].sum()

    tokens_per_ticket = (
        total_tokens / len(classified)
        if len(classified) else 0
    )

    print(f"\n{run_name.upper()} RESULTS")
    print("=" * 40)
    print("Tickets processed:", len(results_df))
    print("\nStatus counts:")
    print(results_df["status"].value_counts())

    print("\nClassified tickets:", len(classified))
    print("Correct classifications:", correct)
    print(f"Accuracy: {accuracy:.2%}")
    print("Total tokens:", total_tokens)
    print(f"Tokens per classified ticket: {tokens_per_ticket:.2f}")

    return results_df, accuracy, total_tokens, tokens_per_ticket


# Run zero-shot classification
zero_results, zero_accuracy, zero_tokens, zero_avg_tokens = (
    run_classification(ZERO_SHOT_PROMPT, "zero-shot")
)

# Run few-shot classification
few_results, few_accuracy, few_tokens, few_avg_tokens = (
    run_classification(FEW_SHOT_PROMPT, "few-shot")
)


print("\nZERO-SHOT VS FEW-SHOT COMPARISON")
print("--------------------------------")
print(f"Zero-shot accuracy: {zero_accuracy:.2%}")
print(f"Few-shot accuracy: {few_accuracy:.2%}")
print("Zero-shot total tokens:", zero_tokens)
print("Few-shot total tokens:", few_tokens)
print("Extra tokens for few-shot:", few_tokens - zero_tokens)


# Zero-shot confusion table
zero_classified = zero_results[
    zero_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

confusion_table = pd.crosstab(
    zero_classified["true_category"],
    zero_classified["category"],
    rownames=["Actual"],
    colnames=["Predicted"]
)

print("\nZERO-SHOT CONFUSION TABLE")
print("-------------------------")
print(confusion_table)
# -----------------------------------------
# STEP 6: TOKEN BUDGET CONTROL
# -----------------------------------------

def run_budget_classification(system_prompt, token_budget):
    mock_llm.reset()

    results = []
    running_tokens = 0

    for _, ticket in processed_df.iterrows():

        if ticket["status"] != "ready":
            results.append({
                "ticket_id": ticket["ticket_id"],
                "customer_id": ticket["customer_id"],
                "category": "",
                "status": ticket["status"],
                "tokens_used": 0
            })
            continue

        # Check budget before making a model call
        if running_tokens >= token_budget:
            results.append({
                "ticket_id": ticket["ticket_id"],
                "customer_id": ticket["customer_id"],
                "category": "",
                "status": "budget_exhausted",
                "tokens_used": 0
            })
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ticket["redacted_text"]}
        ]

        response = call_model_with_retry(
            messages,
            ticket["ticket_id"]
        )

        response_text = response.choices[0].message.content
        tokens = response.usage.total_tokens

        running_tokens += tokens

        category, status, reason = validate_model_output(
            response_text
        )

        results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "category": category,
            "status": status,
            "tokens_used": tokens
        })

    return pd.DataFrame(results), running_tokens


# Run zero-shot with the required budget
TOKEN_BUDGET = 6000

budget_results, budget_total = run_budget_classification(
    ZERO_SHOT_PROMPT,
    TOKEN_BUDGET
)

# If no tickets were stopped, reduce the budget
if "budget_exhausted" not in budget_results["status"].values:
    TOKEN_BUDGET = 5000

    budget_results, budget_total = run_budget_classification(
        ZERO_SHOT_PROMPT,
        TOKEN_BUDGET
    )

# Save budget demonstration
budget_file = OUTPUT_DIR / "triage_budget_demo.csv"
budget_results.to_csv(budget_file, index=False)

print("\nBUDGET DEMONSTRATION")
print("--------------------")
print("Token budget:", TOKEN_BUDGET)
print("Tokens consumed:", budget_total)
print("\nStatus counts:")
print(budget_results["status"].value_counts())
print("\nSaved:", budget_file)
# -----------------------------------------
# STEP 7: FINAL TRIAGE AND ROUTING
# -----------------------------------------

# Load customer segments from Part B
segments_file = OUTPUT_DIR / "customer_segments.csv"
segments = pd.read_csv(segments_file)

# Create customer lookup
segment_lookup = segments.set_index("customer_id").to_dict("index")

final_results = []

for _, result in few_results.iterrows():

    customer_id = result["customer_id"]
    category = result["category"]
    status = result["status"]

    # Find customer segment
    customer_info = segment_lookup.get(customer_id, {})

    segment_name = customer_info.get("segment_name", "")

    is_premium = str(
        customer_info.get("is_premium", False)
    ).strip().lower() in ["true", "1", "yes"]

    # Default routing
    priority = "NORMAL"
    route = "support_queue"

    # Human review for unsafe or unclassified tickets
    if status in [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]:
        route = "human_review"

    elif category == "fraud":
        route = "security_team"
        priority = "HIGH"

    elif is_premium:
        route = "premium_desk"
        priority = "HIGH"

    final_results.append({
        "ticket_id": result["ticket_id"],
        "customer_id": customer_id,
        "status": status,
        "category": category,
        "segment_name": segment_name,
        "priority": priority,
        "route": route,
        "tokens_used": result["tokens_used"]
    })


final_df = pd.DataFrame(final_results)

# Exact required column order
final_df = final_df[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

# Save final output
final_file = OUTPUT_DIR / "triage_results.csv"
final_df.to_csv(final_file, index=False)

print("\nFINAL TRIAGE RESULTS")
print("--------------------")
print("Total tickets:", len(final_df))

print("\nRouting summary:")
print(final_df["route"].value_counts())

print("\nPriority summary:")
print(final_df["priority"].value_counts())

print("\nStatus summary:")
print(final_df["status"].value_counts())

print("\nFirst 10 final records:")
print(final_df.head(10).to_string(index=False))

print("\nSaved:", final_file)

# Final checks
assert len(final_df) == 164
assert list(final_df.columns) == [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

print("\nALL FINAL TRIAGE CHECKS PASSED!")