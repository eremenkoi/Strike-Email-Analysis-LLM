#!/usr/bin/env python3
import json
import os
import subprocess
import re
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
import html

# === CONFIGURATION ===
TRANSCRIPTS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\conversation_transcripts_with_attachments.json"
OUTPUT_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\quote_conversations_filtered_full.json"
EXTRACTION_OUTPUT_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\extracted_info.json"
BATCH_CHECKPOINT_INTERVAL = 100  # Save checkpoint every 100 threads processed
FORCE_RESET = FALSE   # Set to True to start fresh and remove previous output

# Set up logging
logging.basicConfig(
    filename="processing.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Processing started.")

# === CLEANING FUNCTIONS ===
def clean_text(text):
    """Remove HTML tags and unescape HTML entities."""
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def truncate_text(text, max_words=100):
    """Clean and truncate text to a maximum number of words."""
    text = clean_text(text)
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# === EXTRACTION FUNCTION ===
def extract_po_invoice(transcript):
    """
    Extract purchase order and invoice numbers from the transcript.
    Looks for patterns like "PO: 12345" or "Invoice: INV1234".
    """
    combined_text = " ".join(clean_text(msg.get('body', "")) for msg in transcript)
    # Pattern to capture PO numbers (alphanumeric, may include dashes)
    po_match = re.search(r'(?:PO(?:\s*[:\-\.]?\s*))([\w-]+)', combined_text, re.IGNORECASE)
    invoice_match = re.search(r'(?:invoice(?:\s*[:\-\.]?\s*))([\w-]+)', combined_text, re.IGNORECASE)
    return {
        "po_number": po_match.group(1) if po_match else None,
        "invoice_number": invoice_match.group(1) if invoice_match else None
    }

# === HELPER FUNCTIONS ===
def load_transcripts(file_path):
    """Load transcript JSON from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_results(results, file_path):
    """Save results to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Checkpoint saved to {file_path}")
    print(f"💾 Checkpoint saved to {file_path}")

def is_relevant_product(transcript):
    """
    Determine if a transcript is about a relevant product based on keywords.
    Based on Strike's product range: phone cases, rugged cases, cradles, chargers, mounts,
    signal repeaters, antennas, and mobile signal solutions.
    """
    keywords = [
        "cradle", "phone case", "charging cable", "mount", "charger", "phone holder",
        "rugged case", "signal repeater", "antenna", "mobile signal"
    ]
    combined_text = " ".join(clean_text(msg.get('body', "")) for msg in transcript).lower()
    return any(kw in combined_text for kw in keywords)

def is_spam(transcript):
    """
    Determine if a transcript is likely spam.
    Common spam keywords include non-business topics like flights, birthdays, coupons, etc.
    """
    spam_keywords = [
        "qantas", "flight", "birthday", "party", "microsoft 365", "website redesign",
        "mobile app", "coupon", "unsubscribe", "preferences", "travel", "booking", "discount"
    ]
    combined_text = " ".join(clean_text(msg.get('body', "")) for msg in transcript).lower()
    return any(kw in combined_text for kw in spam_keywords)

def build_prompt(thread_id, transcript):
    """Construct the prompt for the LLM from the transcript."""
    conversation_text = ""
    for msg in transcript:
        body = truncate_text(msg.get('body', ""), max_words=100)
        attachment = truncate_text(msg.get('attachment_text', ""), max_words=100)
        date = msg.get('date', "Unknown date")
        sender = msg.get('sender', "Unknown sender")
        conversation_text += (
            f"Date: {date}\n"
            f"Sender: {sender}\n"
            f"Message: {body}\n"
            f"Attachment Text: {attachment}\n"
            "-----\n"
        )

    prompt = (
        "You are an experienced sales strategist at strike.com.au. Based solely on the transcript below, "
        "classify the conversation into one of the following categories:\n\n"
        "1. Quote-related: Direct inquiries or negotiations for purchasing our products.\n"
        "2. Quote-adjacent: Conversations that mention quotes but are not direct negotiations.\n"
        "3. Non-quote adjacent: Conversations unrelated to sales (e.g., spam, notifications).\n\n"
        "IMPORTANT: Do not ask for additional information or request the transcript. "
        "Your response MUST be a single sentence that starts with one of these labels (exactly 'Quote-related:', "
        "'Quote-adjacent:' or 'Non-quote adjacent:') followed by a brief explanation. Do not output any questions.\n\n"
        "Transcript:\n"
        f"{conversation_text}\n\n"
        "Classification:"
    )
    return {"id": thread_id, "prompt": prompt}

def postprocess_output(raw_output):
    """Process the raw output from the model to extract a single relevant line."""
    lines = raw_output.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.lower().startswith("you are") and "classification:" not in stripped.lower():
            return stripped
    return raw_output.strip()

def run_gemma3(prompt_text):
    """
    Runs the Gemma3:12b model via Ollama using a subprocess call.
    The prompt is passed via standard input, with explicit UTF-8 encoding.
    """
    command = ["ollama", "run", "gemma3:12b"]
    logger.info("Running command: " + " ".join(command))
    try:
        result = subprocess.run(
            command,
            input=prompt_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error("Error running Ollama: " + e.stderr)
        print("Error running Ollama:", e.stderr)
        return None

# === MAIN EXECUTION ===
def main():
    # Force a new run by deleting output files if FORCE_RESET is True
    if FORCE_RESET:
        for file in [OUTPUT_FILE, EXTRACTION_OUTPUT_FILE]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Existing {file} removed for a fresh run.")
                print(f"Existing {file} removed for a fresh run.")

    transcripts = load_transcripts(TRANSCRIPTS_FILE)
    total_threads = len(transcripts)
    print(f"Total conversation threads available: {total_threads}")
    logger.info(f"Total conversation threads available: {total_threads}")

    results = {}
    prompt_data = {}
    for thread_id, transcript in transcripts.items():
        if is_spam(transcript):
            results[thread_id] = "Non-quote adjacent (spam)"
        elif not is_relevant_product(transcript):
            results[thread_id] = "Not quote-related (irrelevant product)"
        else:
            prompt_data[thread_id] = build_prompt(thread_id, transcript)["prompt"]

    print(f"Sending {len(prompt_data)} threads to Gemma3 for classification...")
    logger.info(f"Sending {len(prompt_data)} threads to Gemma3 for classification...")

    extraction_results = {}
    processed_count = 0

    # Process all threads in prompt_data (full run)
    for thread_id, prompt in tqdm(prompt_data.items(), total=len(prompt_data)):
        output = run_gemma3(prompt)
        if output:
            processed = postprocess_output(output)
            # Additionally, extract PO/invoice numbers from the original transcript
            po_invoice = extract_po_invoice(transcripts[thread_id])
            results[thread_id] = {
                "classification": processed,
                "extracted_fields": po_invoice
            }
            extraction_results[thread_id] = results[thread_id]
            logger.info(f"Thread {thread_id} processed: {processed} | PO/Invoice: {po_invoice}")
            print(f"Thread {thread_id} processed: {processed}")
        else:
            results[thread_id] = {"classification": "Error processing", "extracted_fields": {}}
            extraction_results[thread_id] = results[thread_id]
            logger.error(f"Thread {thread_id} error processing.")
        processed_count += 1

        # Batch checkpointing every BATCH_CHECKPOINT_INTERVAL threads
        if processed_count % BATCH_CHECKPOINT_INTERVAL == 0:
            save_results(results, OUTPUT_FILE)
            save_results(extraction_results, EXTRACTION_OUTPUT_FILE)
            logger.info(f"Batch checkpoint: {processed_count} threads processed.")

    # Final save after processing
    save_results(results, OUTPUT_FILE)
    save_results(extraction_results, EXTRACTION_OUTPUT_FILE)
    logger.info("🎉 All conversations processed and saved.")
    print("🎉 All conversations processed and saved.")

if __name__ == "__main__":
    main()
