#!/usr/bin/env python3
import json
from transformers import pipeline

# Set the path to your locally downloaded Llama 2 model
LOCAL_MODEL_PATH = r"C:\Users\ierem\PycharmProjects\pythonProject\good_model"


# Load conversation transcripts from JSON file
def load_transcripts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Helper function to truncate text to a maximum number of words
def truncate_text(text, max_words=100):
    words = text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return text


# Check if the conversation mentions our target products (relevant for quotes)
def is_relevant_product(transcript):
    keywords = ["cradle", "phone case", "charging cable", "mount", "charger", "phone holder"]
    combined_text = " ".join(msg['body'] for msg in transcript).lower()
    for kw in keywords:
        if kw in combined_text:
            return True
    return False


# Check if the conversation contains spam/unrelated keywords
def is_spam(transcript):
    spam_keywords = [
        "qantas", "flight", "birthday", "party", "microsoft 365", "website redesign",
        "mobile app", "coupon", "unsubscribe", "preferences", "travel", "booking", "discount"
    ]
    combined_text = " ".join(msg['body'] for msg in transcript).lower()
    for kw in spam_keywords:
        if kw in combined_text:
            return True
    return False


def build_classification_prompt(transcript):
    """
    Build a prompt that instructs the LLM to classify the conversation transcript into one of three categories:
      1. Quote-related
      2. Quote-adjacent
      3. Non-quote adjacent
    For each message, only the first 100 words of the email body and attachment text are used.
    """
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
        "You are an experienced sales strategist at strike.com.au. Your task is to classify the following conversation transcript "
        "into one of these three categories:\n\n"
        "1. Quote-related: Direct inquiries or negotiations for purchasing products we sell (e.g., cradles, phone cases, charging cables, mounts).\n"
        "2. Quote-adjacent: Conversations related to quotes (e.g., returns, refunds, general purchase inquiries) but not direct purchase negotiations.\n"
        "3. Non-quote adjacent: Conversations that are unrelated to our sales, such as spam, system notifications, team updates, or unrelated promotions.\n\n"
        "Answer in one sentence with your classification (choose one of the three) and a brief explanation.\n\n"
        "Use only the content provided below (each email is truncated to the first 100 words) to make your determination.\n\n"
        "Transcript:\n"
        f"{conversation_text}\n\n"
        "Classification:"
    )
    return prompt


def classify_transcript(prompt, llm, max_new_tokens=100):
    result = llm(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, truncation=True)
    return result[0]["generated_text"]


def main():
    transcripts_file = "C:/Users/ierem/PycharmProjects/pythonProject/conversation_transcripts_with_attachments.json"
    transcripts = load_transcripts(transcripts_file)

    print("Loading Llama 2 model...")
    llm = pipeline("text-generation", model=LOCAL_MODEL_PATH, device_map="auto")

    filtered_results = {}
    total_threads = len(transcripts)
    print(f"Total conversation threads available: {total_threads}")

    # Process only the first 10 threads for testing
    for idx, (thread_id, transcript) in enumerate(transcripts.items(), start=1):
        if idx > 10:
            break

        # First check if the transcript is spam
        if is_spam(transcript):
            classification = "Non-quote adjacent (spam)"
        # Next, check if it mentions our target products
        elif not is_relevant_product(transcript):
            classification = "Not quote-related (irrelevant product)"
        else:
            prompt = build_classification_prompt(transcript)
            classification = classify_transcript(prompt, llm)

        filtered_results[thread_id] = classification
        print(f"✅ Processed thread {idx}/10: {classification.strip()}")

        # Save a checkpoint every 5 threads
        if idx % 5 == 0:
            checkpoint_file = "C:/Users/ierem/PycharmProjects/pythonProject/quote_conversations_filtered_test.json"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(filtered_results, f, indent=2)
            print(f"💾 Checkpoint saved at {idx} threads.")

    # Final save after 10 threads
    output_file = "C:/Users/ierem/PycharmProjects/pythonProject/quote_conversations_filtered_test.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(filtered_results, f, indent=2)
    print(f"🎉 Final filtered results saved to {output_file}")


if __name__ == "__main__":
    main()
