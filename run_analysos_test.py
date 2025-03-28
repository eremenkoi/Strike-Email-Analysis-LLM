#!/usr/bin/env python3
import json
import os
from transformers import pipeline

# Set the path to your locally downloaded Falcon 7B model
LOCAL_MODEL_PATH = "C:/Users/ierem/PycharmProjects/pythonProject/models-tiiuae--falcon-7b-instruct"


# Load conversation transcripts from JSON file
def load_transcripts(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Build the prompt for the LLM based on the structured quote analysis requirements
def build_prompt(transcript):
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
        "You are an experienced sales strategist at strike.com.au. We are only interested in analyzing email threads "
        "where strike.com.au is the seller. Please ignore or mark as out of scope any emails where strike.com.au is the buyer "
        "or is not providing a quote.\n\n"
        "Given the conversation transcript below, your tasks are:\n\n"
        "1. **Identify & Focus**\n"
        "   - Confirm whether this thread involves strike.com.au acting as the seller. If not, mark it as “Out of Scope”.\n"
        "   - If yes, proceed with the analysis.\n\n"
        "2. **Categorize the Customer**\n"
        "   - **Industry:** Which industry do they operate in?\n"
        "   - **Size (e.g., SMB, Mid-market, Enterprise):** Based on employees, revenue, or context clues.\n"
        "   - **Maturity (e.g., startup, established, scale-up):** Approximate company age/stage from the conversation.\n\n"
        "3. **Quote Details**\n"
        "   - **Quote Size:** Small, medium, large, or exact monetary value if available.\n"
        "   - **Urgency:** Low, medium, high—did they mention tight deadlines or immediate needs?\n"
        "   - **Complexity:** Standard vs. highly customized.\n"
        "   - **Necessity:** Is the product/solution “nice-to-have” or “must-have”?\n"
        "   - **Initial vs. Final Price:** Were there price changes during negotiation?\n"
        "   - **Negotiation Process:** Who negotiated? Were discounts or competing quotes mentioned?\n"
        "   - **History with Client:** Is there an existing relationship?\n\n"
        "4. **Conversation Dynamics**\n"
        "   - **Initiation:** How was the quote process initiated?\n"
        "   - **Selling Strategies:** Which methods were used by strike.com.au (e.g., value emphasis, demos, discounts, trials)?\n"
        "   - **Emotional Sentiment:** What is the tone on both sides (positive, neutral, frustrated, enthusiastic, etc.)?\n"
        "   - **Response Times:** Were responses timely? Any follow-ups or reminders?\n"
        "   - **Key Objections/Pain Points:** Any concerns about budget, features, timing, etc.?\n\n"
        "5. **Outcome Analysis**\n"
        "   - **Deal Outcome:** Did we win or lose the deal? Summarize how this was determined.\n"
        "   - **Reasons:** Factors such as pricing, features, urgency, competitor offerings, or relationship dynamics.\n"
        "   - **Quote Adjustments:** Any revisions or changes in terms?\n\n"
        "6. **Recommendations**\n"
        "   - Provide structured recommendations for improving future sales/quote interactions (e.g., better customer identification, negotiation strategies, response times, relationship building).\n\n"
        "### Format of the Output\n"
        "Provide a clear, structured analysis. Mark any section as “N/A” or “Out of Scope” where appropriate.\n\n"
        "Transcript:\n"
        f"{conversation_text}\n\n"
        "Analysis:"
    )
    return prompt


# Run the prompt through the LLM and generate a response
def analyze_transcript(prompt, llm, max_length=1400):
    result = llm(prompt, max_length=max_length, do_sample=True, temperature=0.7)
    return result[0]["generated_text"]


def main():
    # Load JSON file
    transcripts_file = "C:/Users/ierem/PycharmProjects/pythonProject/conversation_transcripts_with_attachments.json"
    transcripts = load_transcripts(transcripts_file)

    # Load the Falcon 7B Instruct model from the local path
    print("Loading Falcon 7B Instruct model...")
    llm = pipeline("text-generation", model=LOCAL_MODEL_PATH, device_map="auto")

    # Prepare to save results
    analysis_results = {}
    output_file = "C:/Users/ierem/PycharmProjects/pythonProject/analysis_results_test.json"

    total_threads = len(transcripts)
    print(f"Total conversation threads available: {total_threads}")

    # **Run only the first 10 conversations for testing**
    for idx, (thread_id, transcript) in enumerate(transcripts.items(), start=1):
        if idx > 10:
            break  # Stop after 10 threads

        prompt = build_prompt(transcript)
        analysis = analyze_transcript(prompt, llm)
        analysis_results[thread_id] = analysis

        print(f"Processed thread {idx}/10")

        # Save results after every 5 threads
        if idx % 5 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(analysis_results, f, indent=2)
            print(f"Checkpoint saved at {idx} threads.")

    # Final save after 10 threads
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    print(f"Final analysis results saved to {output_file}")


if __name__ == "__main__":
    main()
