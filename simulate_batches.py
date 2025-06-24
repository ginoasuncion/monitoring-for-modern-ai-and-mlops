import os
import time
import json
import pandas as pd
import openai
from prometheus_client import Gauge, start_http_server
from dotenv import load_dotenv

# Load environment variables from .env (if using)
load_dotenv()

# Set API key from env var
openai.api_key = os.getenv("OPENAI_API_KEY")

# Use gpt-3.5-turbo for faster/cheaper responses
OPENAI_MODEL = "gpt-3.5-turbo"

# Truncate transcripts to avoid hitting token limits
MAX_INPUT_CHARS = 3000

# Prometheus metrics
accuracy = Gauge('summary_factual_accuracy', 'Factual Accuracy', ['product'])
coverage = Gauge('summary_coverage', 'Coverage', ['product'])
coherence = Gauge('summary_coherence', 'Coherence', ['product'])
conciseness = Gauge('summary_conciseness', 'Conciseness', ['product'])

def generate_summary(transcript):
    transcript = transcript[:MAX_INPUT_CHARS]
    prompt = f"Summarize the following product video transcript:\n\n{transcript}"
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

def judge_summary(transcript, summary):
    transcript = transcript[:MAX_INPUT_CHARS]
    prompt = f"""
Evaluate the summary based on the transcript below.

Transcript:
{transcript}

Summary:
{summary}

Score from 0 to 5:
- Factual accuracy
- Coverage
- Coherence
- Conciseness

Return JSON:
{{"Factual accuracy": X, "Coverage": X, "Coherence": X, "Conciseness": X}}
"""
    response = openai.ChatCompletion.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return json.loads(response['choices'][0]['message']['content'].strip())

def log_metrics(product, scores):
    accuracy.labels(product).set(scores["Factual accuracy"])
    coverage.labels(product).set(scores["Coverage"])
    coherence.labels(product).set(scores["Coherence"])
    conciseness.labels(product).set(scores["Conciseness"])

def process_batch(batch_path):
    df = pd.read_csv(batch_path)
    for _, row in df.iterrows():
        product = row["product"]
        transcript = row["concatenated_text"]

        print(f"Processing: {product}")
        try:
            summary = generate_summary(transcript)
            scores = judge_summary(transcript, summary)
            log_metrics(product, scores)
            print(f"✅ {product} metrics: {scores}")
        except Exception as e:
            print(f"❌ Error with {product}: {e}")

        time.sleep(3)  # delay per product to avoid rate limits

def main():
    start_http_server(8000)
    batch_files = sorted([f for f in os.listdir("data") if f.startswith("batch_")])

    for batch_file in batch_files:
        print(f"\n=== Processing {batch_file} ===\n")
        process_batch(os.path.join("data", batch_file))
        print(f"✅ Finished {batch_file} — waiting before next batch...\n")
        time.sleep(10) 

if __name__ == "__main__":
    main()

