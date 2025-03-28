#!/usr/bin/env python3
import json
from transformers import pipeline

# Set the path to your locally downloaded Llama 2 model
LOCAL_MODEL_PATH = r"C:\Users\ierem\PycharmProjects\pythonProject\good_model"


# Load conversation transcripts from JSON file
def load_transcripts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Helper: Truncate text to a maximum number of words
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


# Check if the conversation appears to be spam/unrelated
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

      1. Quote-related: Direct inquiries or negotiations for purchasing products we sell.
      2. Quote-adjacent: Conversations about issues related to quotes (e.g., returns, refunds, general inquiries).
      3. Non-quote adjacent: Conversations unrelated to our sales (e.g., spam, system notifications, unrelated promotions).

    Use only the first 100 words of each message's body and attachment text.
    Output only one sentence with your classification and a brief explanation.
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
        "You are an experienced sales strategist at strike.com.au. Classify the following conversation transcript "
        "into one of the following three categories:\n\n"
        "1. Quote-related: Direct inquiries or negotiations for purchasing products we sell (e.g., cradles, phone cases, charging cables, mounts).\n"
        "2. Quote-adjacent: Conversations related to quotes, such as returns, refunds, or general purchase inquiries, but not direct negotiations.\n"
        "3. Non-quote adjacent: Conversations that are unrelated to our sales (e.g., spam, system notifications, team updates, or unrelated promotions).\n\n"
        "Please output only one sentence with your classification (choose one of the three) and a brief explanation.\n\n"
        "Use only the content provided below (each email is truncated to the first 100 words) to make your determination.\n\n"
        "Transcript:\n"
        f"{conversation_text}\n\n"
        "Classification:"
    )
    return prompt


def postprocess_classification(raw_output):
    """
    Post-process the LLM output to remove any echoed prompt instructions and return only the final answer.
    We'll assume the final answer is the first non-empty line that doesn't repeat the prompt.
    """
    lines = raw_output.split("\n")
    for line in lines:
        stripped = line.strip()
        # If the line is non-empty and doesn't start with common prompt fragments, use it.
        if stripped and not stripped.lower().startswith("you are") and not stripped.lower().startswith(
                "given") and "classification:" not in stripped.lower():
            return stripped
    return raw_output.strip()


def classify_transcript(prompt, llm, max_new_tokens=100):
    result = llm(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, truncation=True)
    raw_output = result[0]["generated_text"]
    return postprocess_classification(raw_output)


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

        # First check if the conversation is spam
        if is_spam(transcript):
            classification = "Non-quote adjacent (spam)"
        # Next, check if it mentions our relevant products
        elif not is_relevant_product(transcript):
            classification = "Not quote-related (irrelevant product)"
        else:
            prompt = build_classification_prompt(transcript)
            classification = classify_transcript(prompt, llm)

        filtered_results[thread_id] = classification
        print(f"✅ Processed thread {idx}/10: {classification}")

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
