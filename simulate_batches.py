import csv
import json
import os
import time
import sys
import openai
import tiktoken
from dotenv import load_dotenv
from statistics import mean
from prometheus_client import start_http_server, Gauge

# --- Load environment variables ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Token limit for input text (e.g., 2000 tokens ≈ safe for gpt-4 w/ response) ---
MAX_TOKENS = 2000
ENCODER = tiktoken.encoding_for_model("gpt-4")

# --- Allow large fields in CSV ---
csv.field_size_limit(sys.maxsize)

# --- Prometheus: batch-level metrics ---
FA_GAUGE = Gauge("batch_factual_accuracy", "Avg factual accuracy", ["batch_id"])
COV_GAUGE = Gauge("batch_coverage", "Avg coverage", ["batch_id"])
COH_GAUGE = Gauge("batch_coherence", "Avg coherence", ["batch_id"])
CONC_GAUGE = Gauge("batch_conciseness", "Avg conciseness", ["batch_id"])

# --- Prometheus: per-product metrics ---
PRODUCT_FA = Gauge("product_factual_accuracy", "Factual accuracy per product", ["product", "batch_id"])
PRODUCT_COV = Gauge("product_coverage", "Coverage per product", ["product", "batch_id"])
PRODUCT_COH = Gauge("product_coherence", "Coherence per product", ["product", "batch_id"])
PRODUCT_CONC = Gauge("product_conciseness", "Conciseness per product", ["product", "batch_id"])

# --- Optional local metric tracker ---
class MetricLogger:
    def __init__(self, name):
        self.name = name
        self.store = {}

    def log(self, product, batch, score):
        self.store[(product, batch)] = score

    def remove(self, product, batch):
        self.store.pop((product, batch), None)

factual_accuracy = MetricLogger("Factual accuracy")
coverage = MetricLogger("Coverage")
coherence = MetricLogger("Coherence")
conciseness = MetricLogger("Conciseness")

# --- Load CSV file ---
def load_batch_data(batch_file):
    with open(batch_file, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# --- Truncate text by token count ---
def truncate_by_tokens(text, max_tokens):
    tokens = ENCODER.encode(text)
    return ENCODER.decode(tokens[:max_tokens])

# --- Evaluate product text using GPT-4 ---
def evaluate_with_llm(product, description):
    # Token-trim the description
    trimmed_description = truncate_by_tokens(description, MAX_TOKENS)
    print(f"\n📝 Evaluating: {product}")
    print(f"📄 Trimmed preview: {trimmed_description[:100]}...")

    prompt = f"""
You are an expert evaluator. Score the following product description on a scale from 1 to 5 for:
- Factual accuracy
- Coverage
- Coherence
- Conciseness

Return your response strictly as a JSON object with these keys.

Product: {product}
Description: {trimmed_description}
"""

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response["choices"][0]["message"]["content"]
    print(f"🔍 Raw LLM response:\n{content}\n")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse response. Returning default metrics.")
        return {
            "Factual accuracy": 1,
            "Coverage": 1,
            "Coherence": 1,
            "Conciseness": 1
        }

# --- Process a single batch file ---
def process_batch(batch_file, batch_id):
    data = load_batch_data(batch_file)
    scores_fa, scores_cov, scores_coh, scores_conc = [], [], [], []

    for entry in data:
        product = entry.get("product", "Unknown")
        description = entry.get("concatenated_text", "").strip()

        if not description:
            print(f"⚠️ Skipping {product}: empty concatenated_text.")
            continue

        try:
            metrics = evaluate_with_llm(product, description)
            print(f"✅ {product} metrics: {metrics}")

            # Store metrics
            factual_accuracy.log(product, batch_id, metrics["Factual accuracy"])
            coverage.log(product, batch_id, metrics["Coverage"])
            coherence.log(product, batch_id, metrics["Coherence"])
            conciseness.log(product, batch_id, metrics["Conciseness"])

            scores_fa.append(metrics["Factual accuracy"])
            scores_cov.append(metrics["Coverage"])
            scores_coh.append(metrics["Coherence"])
            scores_conc.append(metrics["Conciseness"])

            # Prometheus: product-level
            PRODUCT_FA.labels(product=product, batch_id=batch_id).set(metrics["Factual accuracy"])
            PRODUCT_COV.labels(product=product, batch_id=batch_id).set(metrics["Coverage"])
            PRODUCT_COH.labels(product=product, batch_id=batch_id).set(metrics["Coherence"])
            PRODUCT_CONC.labels(product=product, batch_id=batch_id).set(metrics["Conciseness"])

        except Exception as e:
            print(f"❌ Error processing {product}: {e}")
            factual_accuracy.remove(product, batch_id)
            coverage.remove(product, batch_id)
            coherence.remove(product, batch_id)
            conciseness.remove(product, batch_id)

        time.sleep(1.5)

    # --- Per-batch summary ---
    print(f"\n📊 Batch {batch_id} Summary:")
    if scores_fa:
        fa_avg = mean(scores_fa)
        cov_avg = mean(scores_cov)
        coh_avg = mean(scores_coh)
        conc_avg = mean(scores_conc)

        print(f"  - Factual accuracy: {fa_avg:.2f}")
        print(f"  - Coverage:         {cov_avg:.2f}")
        print(f"  - Coherence:        {coh_avg:.2f}")
        print(f"  - Conciseness:      {conc_avg:.2f}")

        FA_GAUGE.labels(batch_id=batch_id).set(fa_avg)
        COV_GAUGE.labels(batch_id=batch_id).set(cov_avg)
        COH_GAUGE.labels(batch_id=batch_id).set(coh_avg)
        CONC_GAUGE.labels(batch_id=batch_id).set(conc_avg)
    else:
        print("  No valid entries in this batch.")

    print("-" * 40)

# --- Main ---
if __name__ == "__main__":
    start_http_server(8000)
    print("🚀 Prometheus metrics available at http://localhost:8000/metrics")

    batch_files = [
        "data/batch_1.csv",
        "data/batch_2.csv",
        "data/batch_3.csv",
        "data/batch_4.csv"
    ]

    for i, batch_file in enumerate(batch_files, start=1):
        print(f"\n=== Processing {batch_file} ===")
        process_batch(batch_file, str(i))
        print("⏳ Waiting 10s before next batch...\n")
        time.sleep(10)

