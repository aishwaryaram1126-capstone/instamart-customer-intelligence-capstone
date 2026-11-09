import sys
import json
import re
import time
import uuid
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"
DB_FILE = ROOT / "data" / "instamart.db"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)

# -----------------------------------------
# STEP 1: LOAD AND CHUNK POLICY DOCUMENTS
# -----------------------------------------

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10

policy_chunks = []


def chunk_text(text, source):
    """Split text into 40-word chunks with 10-word overlap."""

    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]

        chunks.append({
            "source": source,
            "text": " ".join(chunk_words)
        })

        if end >= len(words):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# Read all policy files in alphabetical order
policy_files = sorted(POLICY_DIR.glob("*.txt"))

print("PART D - POLICY RAG ASSISTANT")
print("--------------------------------")
print("Policy documents found:", len(policy_files))

for file_path in policy_files:
    text = file_path.read_text(encoding="utf-8")

    chunks = chunk_text(text, file_path.name)
    policy_chunks.extend(chunks)

    print(
        f"{file_path.name}: "
        f"{len(chunks)} chunks"
    )

print("\nTotal policy chunks:", len(policy_chunks))

# Verify assignment requirements
assert len(policy_files) == 5
assert len(policy_chunks) == 15

print("\nALL POLICY CHUNKING CHECKS PASSED!")
# -----------------------------------------
# STEP 2: TF-IDF POLICY SEARCH
# -----------------------------------------

# Convert policy chunks into TF-IDF vectors
vectorizer = TfidfVectorizer()

chunk_texts = [
    chunk["text"] for chunk in policy_chunks
]

tfidf_matrix = vectorizer.fit_transform(chunk_texts)


def retrieve(query, k=3):
    """Retrieve the k most relevant policy chunks."""

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    top_indices = np.argsort(scores)[::-1][:k]

    results = []

    for index in top_indices:
        results.append({
            "source": policy_chunks[index]["source"],
            "score": float(scores[index]),
            "text": policy_chunks[index]["text"]
        })

    return results


# Test policy retrieval
test_query = "refund policy late delivery"

search_results = retrieve(test_query, k=3)

print("\nTF-IDF POLICY SEARCH")
print("--------------------")
print("Test query:", test_query)

for rank, result in enumerate(search_results, start=1):
    print(f"\nResult {rank}")
    print("Source:", result["source"])
    print("Similarity score:", round(result["score"], 4))
    print("Text:", result["text"])

# Assignment check
assert search_results[0]["source"] == "refund_policy.txt"

print("\nTOP RESULT CHECK PASSED!")
# -----------------------------------------
# STEP 3: RAG POLICY ANSWER FUNCTION
# -----------------------------------------

def answer(question):
    """Answer a policy question using retrieved context."""

    retrieved = retrieve(question, k=3)

    # Build context using the required format
    context = "\n".join(
        f"[{item['source']}]\n{item['text']}"
        for item in retrieved
    )

    user_content = (
        "CONTEXT:\n"
        + context
        + "\nQUESTION:\n"
        + question
    )

    system_prompt = """
You are a POLICY ASSISTANT.

Answer the question using only the supplied policy context.
Do not invent or assume policy rules.
If the answer cannot be found in the context,
respond beginning exactly with:
I could not find
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    response = mock_llm.chat(messages)

    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(
        item["source"] for item in retrieved
    ))

    return {
        "answer": answer_text,
        "sources": sources
    }


# Test the RAG assistant
test_question = "What happens if an order is delivered after 45 minutes?"

rag_result = answer(test_question)

print("\nRAG POLICY ANSWER")
print("-----------------")
print("Question:", test_question)
print("Answer:", rag_result["answer"])
print("Sources:", rag_result["sources"])

print("\nRAG ANSWER FUNCTION CREATED!")
# -----------------------------------------
# STEP 4: INSPECT POLICY EVALUATION DATA
# -----------------------------------------

EVAL_FILE = ROOT / "data" / "raw" / "policy_eval.csv"

evaluation_df = pd.read_csv(EVAL_FILE)

print("\nPOLICY EVALUATION DATASET")
print("------------------------")
print("Total questions:", len(evaluation_df))

print("\nColumn names:")
print(evaluation_df.columns.tolist())

print("\nFirst 5 evaluation records:")
print(evaluation_df.head().to_string(index=False))

print("\nEvaluation dataset loaded successfully!")

assert len(evaluation_df) == 14
# -----------------------------------------
# STEP 5: EVALUATE RAG ASSISTANT
# -----------------------------------------

def evaluate_rag(eval_data, label):
    """Evaluate document retrieval and answer accuracy."""

    doc_correct = 0
    fact_correct = 0
    fallback_correct = 0

    evaluation_results = []

    for _, row in eval_data.iterrows():

        question = str(row["question"])
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"])

        # Retrieve the top 3 policy chunks
        retrieved = retrieve(question, k=3)

        retrieved_sources = [
            item["source"] for item in retrieved
        ]

        result = answer(question)
        answer_text = result["answer"]

        # Identify unanswerable questions
        unanswerable = expected_doc.lower() in [
            "", "none", "nan", "unanswerable"
        ]

        if unanswerable:
            fallback_ok = answer_text.startswith(
                "I could not find"
            )

            if fallback_ok:
                fallback_correct += 1

            doc_ok = None
            fact_ok = fallback_ok

        else:
            doc_ok = expected_doc in retrieved_sources
            fact_ok = expected_fact in answer_text

            if doc_ok:
                doc_correct += 1

            if fact_ok:
                fact_correct += 1

        evaluation_results.append({
            "question": question,
            "expected_doc": expected_doc,
            "retrieved_sources": ", ".join(
                dict.fromkeys(retrieved_sources)
            ),
            "answer": answer_text,
            "document_correct": doc_ok,
            "fact_correct": fact_ok
        })

    results_df = pd.DataFrame(evaluation_results)

    answerable_count = sum(
        str(value).strip().lower()
        not in ["", "none", "nan", "unanswerable"]
        for value in eval_data["expected_doc"]
    )

    unanswerable_count = len(eval_data) - answerable_count

    print(f"\n{label}")
    print("=" * 45)

    print(
        f"Document retrieval: "
        f"{doc_correct}/{answerable_count}"
    )

    print(
        f"Expected fact found: "
        f"{fact_correct}/{answerable_count}"
    )

    print(
        f"Correct fallback: "
        f"{fallback_correct}/{unanswerable_count}"
    )

    return results_df


# Evaluate using 10-word overlap
mock_llm.reset()

overlap_results = evaluate_rag(
    evaluation_df,
    "RAG EVALUATION - 10 WORD OVERLAP"
)


# -----------------------------------------
# REPEAT WITH ZERO WORD OVERLAP
# -----------------------------------------

zero_overlap_chunks = []

for file_path in policy_files:

    text = file_path.read_text(encoding="utf-8")

    words = text.split()

    start = 0

    while start < len(words):

        chunk_words = words[start:start + CHUNK_SIZE]

        zero_overlap_chunks.append({
            "source": file_path.name,
            "text": " ".join(chunk_words)
        })

        start += CHUNK_SIZE


# Temporarily replace the original chunks
original_chunks = policy_chunks
original_vectorizer = vectorizer
original_matrix = tfidf_matrix

policy_chunks = zero_overlap_chunks

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([
    chunk["text"] for chunk in policy_chunks
])

mock_llm.reset()

zero_overlap_results = evaluate_rag(
    evaluation_df,
    "RAG EVALUATION - ZERO WORD OVERLAP"
)


# Restore original 10-word overlap system
policy_chunks = original_chunks
vectorizer = original_vectorizer
tfidf_matrix = original_matrix

print("\nRAG EVALUATION COMPLETED!")
# -----------------------------------------
# STEP 6: REFUND AGENT TOOLS
# -----------------------------------------

import sqlite3


def get_order(order_id):
    """Retrieve order details from SQLite."""

    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return {
                "error": "ORDER_NOT_FOUND",
                "order_id": order_id
            }

        return dict(row)

    finally:
        connection.close()


def search_policy(query):
    """Search the policy documents."""

    return {
        "results": retrieve(query, k=3)
    }


def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund based on late-delivery policy."""

    if float(delivery_minutes) > 45:
        refund = float(order_amount)
    else:
        refund = 0.0

    return {
        "refund_inr": round(refund, 2),
        "eligible": refund > 0
    }


def write_ledger(tool_name, order_id, refund_inr):
    """Append an action to the refund ledger."""

    ledger_file = OUTPUT_DIR / "refund_ledger.jsonl"

    record = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": float(refund_inr)
    }

    with ledger_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "status": "SUCCESS",
        "refund_inr": float(refund_inr),
        "record": record
    }
# -----------------------------------------
# STEP 7: REFUND ACTION TOOLS
# -----------------------------------------

def issue_refund(order_id, refund_inr):
    """Record an approved refund."""
    return write_ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    """Record a refund requiring human approval."""
    return write_ledger("escalate_to_human", order_id, refund_inr)


# -----------------------------------------
# STEP 8: TOOL REGISTRY
# -----------------------------------------

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

print("\nALL FIVE AGENT TOOLS REGISTERED!")
# -----------------------------------------
# STEP 9: CHECKPOINT AND RESUME
# -----------------------------------------

def save_checkpoint(run_id, state):
    """Save agent progress after every tool observation."""
    checkpoint_file = STATE_DIR / f"{run_id}.json"

    with checkpoint_file.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)


def load_checkpoint(run_id):
    """Load a saved agent run."""
    checkpoint_file = STATE_DIR / f"{run_id}.json"

    with checkpoint_file.open("r", encoding="utf-8") as file:
        return json.load(file)


# -----------------------------------------
# STEP 10: TOOL SPECIFICATION
# -----------------------------------------

TOOLS_SPEC = [
    {"type": "function", "function": {"name": name}}
    for name in TOOL_MAP
]


# -----------------------------------------
# STEP 11: REACT AGENT
# -----------------------------------------

def continue_agent(state, max_iterations=6,
                   token_limit=6000, time_limit_seconds=10,
                   crash_after_tools=None):

    started = time.monotonic()
    tool_count = state.get("tool_count", 0)

    while state["iteration"] < max_iterations:

        if time.monotonic() - started >= time_limit_seconds:
            state["status"] = "FALLBACK_TIMEOUT"
            save_checkpoint(state["run_id"], state)
            return state

        if state["token_count"] >= token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            save_checkpoint(state["run_id"], state)
            return state

        try:
            response = mock_llm.chat(
                state["messages"],
                tools=TOOLS_SPEC
            )
        except mock_llm.TransientLLMError:
            try:
                response = mock_llm.chat(
                    state["messages"],
                    tools=TOOLS_SPEC
                )
            except Exception:
                state["status"] = "FALLBACK_ERROR"
                save_checkpoint(state["run_id"], state)
                return state

        state["token_count"] += response.usage.total_tokens
        state["iteration"] += 1

        if state["token_count"] > token_limit:
            state["status"] = "FALLBACK_TOKEN_LIMIT"
            save_checkpoint(state["run_id"], state)
            return state

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            state["answer"] = message.content or ""
            state["status"] = (
                state["answer"].split(":")[0]
                if ":" in state["answer"]
                else "COMPLETED"
            )
            save_checkpoint(state["run_id"], state)
            return state

        for tool_call in tool_calls:
            tool_name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                state["status"] = "FALLBACK_UNKNOWN_TOOL"
                save_checkpoint(state["run_id"], state)
                return state

            result = TOOL_MAP[tool_name](**arguments)

            state["messages"].append({
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(result, default=str)
            })

            tool_count += 1
            state["tool_count"] = tool_count

            # Save progress immediately after each tool observation.
            save_checkpoint(state["run_id"], state)

            if crash_after_tools == tool_count:
                state["status"] = "PAUSED_FOR_RESUME"
                save_checkpoint(state["run_id"], state)
                return state

    state["status"] = "FALLBACK_MAX_ITERATIONS"
    save_checkpoint(state["run_id"], state)
    return state


def run_agent(user_message, scenario_id,
              max_iterations=6,
              token_limit=6000,
              time_limit_seconds=10,
              crash_after_tools=None):

    run_id = f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AGENT. Process refund requests "
                "using the available tools and refund policy."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "iteration": 0,
        "token_count": 0,
        "tool_count": 0,
        "status": "RUNNING",
        "answer": ""
    }

    save_checkpoint(run_id, state)

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_after_tools=crash_after_tools
    )


def resume_agent(run_id, max_iterations=6,
                 token_limit=6000, time_limit_seconds=10):

    state = load_checkpoint(run_id)

    if state["status"] != "PAUSED_FOR_RESUME":
        return state

    state["status"] = "RUNNING"

    return continue_agent(
        state,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds
    )


print("\nREACT AGENT AND CHECKPOINT FUNCTIONS CREATED!")
# -----------------------------------------
# STEP 12: AGENT TEST SCENARIOS
# -----------------------------------------

SCENARIOS = [
    {
        "id": "S1",
        "message": "Please process refund for ORD-100048"
    },
    {
        "id": "S2",
        "message": "Please process refund for ORD-106173"
    },
    {
        "id": "S3",
        "message": "Please process refund for ORD-100145"
    },
    {
        "id": "S4",
        "message": "Please process refund for ORD-102006"
    },
    {
        "id": "S5",
        "message": "Please process refund for ORD-110093"
    },
    {
        "id": "S6",
        "message": "Please process refund for ORD-999999"
    },
    {
        "id": "S7",
        "message": "[LOOP] Please process this refund request"
    }
]


# -----------------------------------------
# STEP 13: RUN ALL SCENARIOS
# -----------------------------------------

print("\nREACT AGENT SCENARIO TESTS")
print("=" * 45)

mock_llm.reset()

scenario_results = []

for scenario in SCENARIOS:
    print(f"\nRunning {scenario['id']}...")

    result = run_agent(
        user_message=scenario["message"],
        scenario_id=scenario["id"]
    )

    print("Status:", result["status"])
    print("Answer:", result.get("answer", ""))

    scenario_results.append({
        "scenario_id": scenario["id"],
        "status": result["status"],
        "answer": result.get("answer", ""),
        "run_id": result["run_id"]
    })


# Save scenario results
results_file = OUTPUT_DIR / "agent_runs.jsonl"

with results_file.open("w", encoding="utf-8") as file:
    for result in scenario_results:
        file.write(json.dumps(result) + "\n")

print("\nALL SEVEN SCENARIOS EXECUTED!")
print("Results saved to outputs/agent_runs.jsonl")
# -----------------------------------------
# STEP 14: CHECKPOINT AND RESUME TEST
# -----------------------------------------

print("\nCHECKPOINT AND RESUME TEST")
print("=" * 40)

mock_llm.reset()

paused_result = run_agent(
    user_message="Please process refund for ORD-100048",
    scenario_id="CHECKPOINT_TEST",
    crash_after_tools=2
)

print("Initial status:", paused_result["status"])
print("Tools completed:", paused_result["tool_count"])

saved_run_id = paused_result["run_id"]

resumed_result = resume_agent(saved_run_id)

print("Resumed status:", resumed_result["status"])
print("Final answer:", resumed_result["answer"])
print("Total tools completed:", resumed_result["tool_count"])

assert paused_result["status"] == "PAUSED_FOR_RESUME"
assert resumed_result["status"] == "REFUND"
assert resumed_result["tool_count"] == 4

print("\nCHECKPOINT AND RESUME TEST PASSED!")
# -----------------------------------------
# STEP 15: TOKEN AND TIMEOUT LIMIT TESTS
# -----------------------------------------

print("\nAGENT SAFETY LIMIT TESTS")
print("=" * 40)

# Test token limit
mock_llm.reset()

token_result = run_agent(
    user_message="Please process refund for ORD-100048",
    scenario_id="TOKEN_TEST",
    token_limit=1
)

print("Token limit result:", token_result["status"])

assert token_result["status"] == "FALLBACK_TOKEN_LIMIT"


# Test timeout
mock_llm.reset()

timeout_result = run_agent(
    user_message="Please process refund for ORD-100048",
    scenario_id="TIMEOUT_TEST",
    time_limit_seconds=0
)

print("Timeout result:", timeout_result["status"])

assert timeout_result["status"] == "FALLBACK_TIMEOUT"

print("\nALL AGENT SAFETY LIMIT TESTS PASSED!")