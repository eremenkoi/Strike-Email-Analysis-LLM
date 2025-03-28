#!/usr/bin/env python3
import json
from transformers import pipeline

# Set the path to your locally downloaded Llama 2 model
LOCAL_MODEL_PATH = r"C:\Users\ierem\PycharmProjects\pythonProject\good_model"


# Load conversation transcripts from JSON file
def load_transcripts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_classification_prompt(transcript):
    """
    Build a prompt that instructs the LLM to classify the conversation transcript
    into one of three categories:

    1. Quote-related: The conversation involves direct inquiries or negotiations
       for purchasing products we sell (e.g., cradles, phone cases, charging cables, mounts).
    2. Quote adjacent: The conversation is about issues related to quotes (such as returns,
       refunds, or general inquiries) but not a direct purchase negotiation.
    3. Non-quote adjacent: The conversation is unrelated to sales, such as spam, system notifications,
       team updates, or general news.

    Answer in one sentence with your classification (choose one of the three) and a brief explanation.
    """
    conversation_text = ""
    for msg in transcript:
        conversation_text += (
            f"Date: {msg['date']}\n"
            f"Sender: {msg['sender']}\n"
            f"Message: {msg['body']}\n"
            f"Attachment Text: {msg['attachment_text']}\n"
            "-----\n"
        )

    prompt = (
        "You are an experienced sales strategist at strike.com.au. Please classify the following conversation transcript "
        "into one of these three categories:\n\n"
        "1. Quote-related: Direct inquiries or negotiations for purchasing products we sell (e.g., cradles, phone cases, charging cables, mounts).\n"
        "2. Quote adjacent: Conversations about issues related to quotes, such as returns, refunds, or general inquiries.\n"
        "3. Non-quote adjacent: Conversations that are unrelated to sales, such as spam, system notifications, team updates, or general news.\n\n"
        "Answer in one sentence with your classification and a brief explanation.\n\n"
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
            break  # Stop after 10 threads

        prompt = build_classification_prompt(transcript)
        classification = classify_transcript(prompt, llm)
        filtered_results[thread_id] = classification

        print(f"✅ Processed thread {idx}/10: {classification.strip()}")

        # Save checkpoint every 5 threads
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
