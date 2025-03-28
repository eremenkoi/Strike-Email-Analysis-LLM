#!/usr/bin/env python3
import json
import os
from transformers import pipeline

# === CONFIGURATION ===
LOCAL_MODEL_PATH = r"C:\Users\ierem\PycharmProjects\pythonProject\good_model"
TRANSCRIPTS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\conversation_transcripts_with_attachments.json"
OUTPUT_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\quote_conversations_filtered_full.json"
CHECKPOINT_INTERVAL = 10

# === HELPERS ===
def truncate_text(text, max_words=100):
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def load_transcripts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(results, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Checkpoint saved to {file_path}")

def is_relevant_product(transcript):
    keywords = ["cradle", "phone case", "charging cable", "mount", "charger", "phone holder"]
    combined_text = " ".join(msg['body'] for msg in transcript).lower()
    return any(kw in combined_text for kw in keywords)

def is_spam(transcript):
    spam_keywords = [
        "qantas", "flight", "birthday", "party", "microsoft 365", "website redesign",
        "mobile app", "coupon", "unsubscribe", "preferences", "travel", "booking", "discount"
    ]
    combined_text = " ".join(msg['body'] for msg in transcript).lower()
    return any(kw in combined_text for kw in spam_keywords)

def build_classification_prompt(transcript):
    conversation_text = ""
    for msg in transcript:
        body = truncate_text(msg['body'], max_words=100)
        attachment = truncate_text(msg['attachment_text'], max_words=100)
        conversation_text += (
            f"Date: {msg['date']}\n"
            f"Sender: {msg['sender']}\n"
            f"Message: {body}\n"
            f"Attachment Text: {attachment}\n"
            "-----\n"
        )

    prompt = (
        "You are an experienced sales strategist at strike.com.au. Classify the following conversation transcript "
        "into one of these three categories:\n\n"
        "1. Quote-related: Direct inquiries or negotiations for purchasing products we sell (e.g., cradles, phone cases, charging cables, mounts).\n"
        "2. Quote-adjacent: Conversations related to quotes, such as returns, refunds, or general purchase inquiries, but not direct negotiations.\n"
        "3. Non-quote adjacent: Conversations that are unrelated to our sales (e.g., spam, system notifications, team updates, or unrelated promotions).\n\n"
        "Please output only one sentence with your classification (choose one of the three) and a brief explanation.\n\n"
        "Transcript:\n"
        f"{conversation_text}\n\n"
        "Classification:"
    )
    return prompt

def postprocess_classification(raw_output):
    lines = raw_output.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("you are") and "classification:" not in stripped.lower():
            return stripped
    return raw_output.strip()

def classify_transcript(prompt, llm, max_new_tokens=100):
    result = llm(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return postprocess_classification(result[0]["generated_text"])

# === MAIN EXECUTION ===
def main():
    transcripts = load_transcripts(TRANSCRIPTS_FILE)
    print(f"Total conversation threads available: {len(transcripts)}")

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            analysis_results = json.load(f)
        print(f"Resuming from {len(analysis_results)} previously processed threads.")
    else:
        analysis_results = {}

    llm = pipeline("text-generation", model=LOCAL_MODEL_PATH, device_map="auto")

    new_processed = 0
    for idx, (thread_id, transcript) in enumerate(transcripts.items(), start=1):
        if thread_id in analysis_results:
            continue

        if is_spam(transcript):
            classification = "Non-quote adjacent (spam)"
        elif not is_relevant_product(transcript):
            classification = "Not quote-related (irrelevant product)"
        else:
            prompt = build_classification_prompt(transcript)
            classification = classify_transcript(prompt, llm)

        analysis_results[thread_id] = classification
        new_processed += 1
        print(f"✅ Processed thread {idx}: {classification}")

        if new_processed % CHECKPOINT_INTERVAL == 0:
            save_results(analysis_results, OUTPUT_FILE)

    save_results(analysis_results, OUTPUT_FILE)
    print("🎉 All conversations processed and saved.")

if __name__ == "__main__":
    main()