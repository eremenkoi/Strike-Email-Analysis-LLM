#!/usr/bin/env python3
import json
import os
import re
import logging
import time
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import html
import concurrent.futures
from datetime import datetime

# Try to use the Ollama client library if available, otherwise fallback to subprocess
try:
    from ollama import Client

    ollama_client = Client()
    USING_CLIENT = True
    print("Using Ollama client library")
except ImportError:
    import subprocess

    USING_CLIENT = False
    print("Ollama client library not found, falling back to subprocess")

# === CONFIGURATION ===
TRANSCRIPTS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\conversation_transcripts_with_attachments.json"
OUTPUT_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\quote_conversations_filtered_enhanced.json"
FULL_DETAILS_OUTPUT_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\extracted_info_full_details_enhanced.json"
FIRST_PASS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\first_pass_filtered_threads.json"
SUMMARY_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\extraction_summary_report.json"
LLM_MODEL = "gemma3:12b"  # Model name for Ollama
BATCH_SIZE = 20  # Number of threads to process in each batch
MAX_WORKERS = 4  # Number of parallel workers (adjust based on your system)
BATCH_CHECKPOINT_INTERVAL = 1  # Save checkpoint after each batch
FORCE_RESET = True  # Set to True to start fresh and remove previous output files

# Set up logging
log_filename = f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Processing started.")


# === CLEANING FUNCTIONS ===
def clean_text(text):
    """Remove HTML tags and unescape HTML entities."""
    if not text or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


def truncate_text(text, max_words=100):
    """Clean and truncate text to a maximum number of words."""
    text = clean_text(text)
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")


# Define common keyword lists at module level for reuse
# Product keywords - expanded with variants and specific product terms
PRODUCT_KEYWORDS = [
    # Phone cases and protection
    "case", "cover", "protection", "screen protector", "rugged", "tough", "shockproof",

    # Mounting solutions
    "cradle", "mount", "holder", "dock", "bracket", "stand",
    "dashboard", "windshield", "suction", "adhesive",

    # Charging solutions
    "charger", "charging", "power", "cable", "adapter", "battery", "portable",
    "car charger", "wireless", "usb", "type-c", "lightning",

    # Signal solutions
    "signal", "antenna", "repeater", "booster", "amplifier", "reception",
    "4g", "5g", "cellular", "mobile", "network", "coverage",

    # Common product identifiers
    "model", "part number", "sku", "product code", "item", "accessory"
]


# === FIRST PASS FILTERING FUNCTIONS ===
def is_relevant_product(transcript):
    """
    Enhanced relevance detection using broader keyword sets and context.
    """

    # Quote-related keywords
    quote_keywords = [
        "quote", "price", "pricing", "cost", "estimate", "proposal",
        "invoice", "purchase order", "po", "order", "buy", "purchase",
        "$", "aud", "usd", "payment", "discount", "offer", "sale",
        "inquiry", "enquiry", "interested in", "looking for", "need"
    ]

    # Combine all text from the transcript
    combined_text = " ".join(clean_text(msg.get('body', "")) for msg in transcript).lower()

    # Check if the conversation mentions both products and quote-related terms
    has_product = any(kw in combined_text for kw in PRODUCT_KEYWORDS)
    has_quote_term = any(kw in combined_text for kw in quote_keywords)

    # If it has both product and quote keywords, it's definitely relevant
    if has_product and has_quote_term:
        return True

    # If it has product keywords but no quote terms, check if it's a substantial conversation
    # (more than 3 messages might indicate ongoing product discussion)
    if has_product and len(transcript) > 3:
        return True

    # If it mentions quote terms but no specific products, it might still be relevant
    if has_quote_term and len(transcript) > 2:
        return True

    # Default to False if none of the above conditions are met
    return False


def is_spam(transcript):
    """
    Enhanced spam detection with more comprehensive patterns.
    """
    # Definite spam indicators
    strong_spam_keywords = [
        "unsubscribe", "click here", "earning potential", "free gift",
        "limited time offer", "lottery", "winner", "congratulations",
        "viagra", "pharmacy", "prescription", "medication"
    ]

    # Non-business topics unlikely to be related to quotes
    off_topic_keywords = [
        "flight", "booking", "holiday", "vacation", "travel", "birthday",
        "party", "wedding", "invitation", "celebration", "condolence",
        "resume", "cv", "job application", "interview", "recruitment",
        "newsletter", "subscription", "preferences", "coupon", "discount code"
    ]

    # Company services unrelated to physical products
    non_product_services = [
        "website redesign", "mobile app", "seo", "marketing campaign",
        "social media", "branding", "logo design", "consulting",
        "training session", "workshop", "seminar", "webinar"
    ]

    combined_text = " ".join(clean_text(msg.get('body', "")) for msg in transcript).lower()

    # If it contains any strong spam indicators, it's definitely spam
    if any(kw in combined_text for kw in strong_spam_keywords):
        return True

    # Check for patterns of off-topic content
    off_topic_count = sum(1 for kw in off_topic_keywords if kw in combined_text)
    if off_topic_count >= 2:  # If multiple off-topic keywords appear, likely spam
        return True

    # Check for non-product service discussions
    non_product_count = sum(1 for kw in non_product_services if kw in combined_text)
    if non_product_count >= 2:  # Multiple mentions of non-product services
        return True

    # Very short conversations with no product keywords are likely spam
    if len(transcript) <= 2 and not any(kw in combined_text for kw in PRODUCT_KEYWORDS):
        return True

    return False


def contains_quote_indicators(transcript):
    """
    Additional function to detect specific quote indicators that strongly suggest
    a quote-related conversation.
    """
    strong_quote_indicators = [
        # Price indicators
        r"\$\d+", r"\d+ dollars", r"\$\d+\.\d+", r"\d+\.\d+ aud", r"\d+\.\d+ usd",

        # Quote request patterns
        r"send.*quote", r"request.*quote", r"quote.*request",
        r"need.*quote", r"provide.*quote", r"quote.*attached",

        # Invoice/PO patterns
        r"invoice #\w+", r"invoice number", r"po #\w+", r"purchase order #\w+",
        r"po number", r"purchase order number",

        # Pricing discussion patterns
        r"cost per unit", r"unit price", r"total cost", r"excluding gst",
        r"including gst", r"plus gst", r"discount.*%", r"\d+% discount"
    ]

    combined_text = " ".join(clean_text(msg.get('body', "")) + " " +
                             clean_text(msg.get('attachment_text', ""))
                             for msg in transcript).lower()

    # Check for strong indicators
    for pattern in strong_quote_indicators:
        if re.search(pattern, combined_text):
            return True

    return False


def has_financial_values(transcript):
    """
    Check if the transcript contains financial values like dollar amounts.
    """
    money_pattern = r'(?:aud|usd)?\s?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s?(?:aud|usd))?'

    combined_text = " ".join(clean_text(msg.get('body', "")) for msg in transcript).lower()

    # Find all financial values
    matches = re.findall(money_pattern, combined_text)

    # Return True if we find more than one financial value (more reliable indicator)
    return len(matches) > 1


def first_pass_analysis(transcript):
    """
    Comprehensive first-pass analysis to classify conversations without using LLM.
    Returns a tuple of (is_potentially_quote_related, confidence, reason)
    """
    # Skip spam immediately
    if is_spam(transcript):
        return (False, 0.9, "Identified as spam")

    # Check for quote indicators (strongest signal)
    if contains_quote_indicators(transcript):
        return (True, 0.9, "Contains explicit quote indicators")

    # Check for relevant products + financial values (strong signal)
    if is_relevant_product(transcript) and has_financial_values(transcript):
        return (True, 0.8, "Mentions relevant products and financial values")

    # Check for relevant products only (moderate signal)
    if is_relevant_product(transcript):
        return (True, 0.6, "Mentions relevant products")

    # Check for financial values only (weak signal)
    if has_financial_values(transcript):
        return (True, 0.4, "Contains financial values")

    # Default - not likely quote related
    return (False, 0.7, "No quote indicators detected")


# === HELPER FUNCTIONS ===
def load_transcripts(file_path):
    """Load transcript JSON from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading transcripts: {str(e)}")
        print(f"Error loading transcripts: {str(e)}")
        return {}


def save_results(results, file_path):
    """Save results to a JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Data saved to {file_path}")
        print(f"💾 Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        print(f"Error saving results: {str(e)}")


# === PROMPT ENGINEERING ===
def build_prompt(thread_id, transcript):
    """Construct an improved prompt for the LLM with examples and clear instructions."""
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
        "extract and output a JSON object with these specifications:\n\n"

        "CLASSIFICATION CRITERIA:\n"
        "- \"Quote-related\": Direct mentions of quotes, pricing, or product orders\n"
        "- \"Quote-adjacent\": Discussions about products that may lead to quotes, but no direct quote request\n"
        "- \"Non-quote adjacent\": Unrelated to purchasing or quoting\n\n"

        "OUTPUT FORMAT:\n"
        "{\n"
        "  \"classification\": string (one of \"Quote-related\", \"Quote-adjacent\", or \"Non-quote adjacent\"),\n"
        "  \"reason\": string (brief justification for classification),\n"
        "  \"client_company\": string or null (company name if mentioned),\n"
        "  \"client_email\": string or null (email address if available),\n"
        "  \"order_total\": number or null (total amount if mentioned),\n"
        "  \"purchase_invoice\": string or null (PO number if available),\n"
        "  \"sales_invoice\": string or null (invoice number if available)\n"
        "}\n\n"

        "EXAMPLE 1:\n"
        "If the transcript mentions \"Thank you for the quote of $1,250 for 5 phone cradles\", output:\n"
        "{\n"
        "  \"classification\": \"Quote-related\",\n"
        "  \"reason\": \"Direct mention of quote with pricing for specific product\",\n"
        "  \"client_company\": null,\n"
        "  \"client_email\": null,\n"
        "  \"order_total\": 1250,\n"
        "  \"purchase_invoice\": null,\n"
        "  \"sales_invoice\": null\n"
        "}\n\n"

        "EXAMPLE 2:\n"
        "If a customer asks \"Can you send me pricing for rugged iPhone cases?\", output:\n"
        "{\n"
        "  \"classification\": \"Quote-adjacent\",\n"
        "  \"reason\": \"Inquiry about pricing but no formal quote yet\",\n"
        "  \"client_company\": null,\n"
        "  \"client_email\": null,\n"
        "  \"order_total\": null,\n"
        "  \"purchase_invoice\": null,\n"
        "  \"sales_invoice\": null\n"
        "}\n\n"

        "NOW ANALYZE THIS TRANSCRIPT:\n"
        f"{conversation_text}\n\n"
        "Return ONLY valid JSON without explanation or markup."
    )
    return {"id": thread_id, "prompt": prompt}


# === LLM INTERACTION ===
def run_gemma3(prompt_text):
    """
    Run the LLM model using either direct API or subprocess.
    """
    start_time = time.time()
    try:
        if USING_CLIENT:
            # Use Ollama Python client
            response = ollama_client.chat(model=LLM_MODEL, messages=[
                {"role": "user", "content": prompt_text}
            ])
            duration = time.time() - start_time
            logger.info(f"LLM query completed in {duration:.2f} seconds")
            return response['message']['content']
        else:
            # Fallback to subprocess
            command = ["ollama", "run", LLM_MODEL]
            logger.info("Running command: " + " ".join(command))
            result = subprocess.run(
                command,
                input=prompt_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True
            )
            duration = time.time() - start_time
            logger.info(f"LLM query completed in {duration:.2f} seconds")
            return result.stdout
    except Exception as e:
        logger.error(f"Error running LLM: {str(e)}")
        print(f"Error running LLM: {str(e)}")
        return None


def postprocess_output(raw_output):
    """
    Enhanced processing of LLM output with better error handling and validation.
    """
    if not raw_output:
        return {
            "classification": "Error",
            "reason": "No output from LLM",
            "client_company": None,
            "client_email": None,
            "order_total": None,
            "purchase_invoice": None,
            "sales_invoice": None,
            "raw_output": None
        }

    # First try to extract JSON if it's embedded in other text
    json_pattern = r'(\{.*\})'
    json_matches = re.findall(json_pattern, raw_output, re.DOTALL)

    if json_matches:
        # Use the largest JSON object found (most complete)
        candidate = max(json_matches, key=len)
        try:
            data = json.loads(candidate)
            # Validate the required fields
            if "classification" not in data:
                data["classification"] = "Error"
                data["reason"] = "Missing classification field"
            return data
        except json.JSONDecodeError:
            pass  # If this fails, fall back to the full text parsing

    # Try to parse the full output as JSON
    try:
        data = json.loads(raw_output.strip())
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e} | Raw output: {raw_output.strip()}")

        # Create our best attempt at parsing the fields
        result = {
            "classification": "Error",
            "reason": "Failed to decode JSON",
            "client_company": None,
            "client_email": None,
            "order_total": None,
            "purchase_invoice": None,
            "sales_invoice": None,
            "raw_output": raw_output.strip()
        }

        # Try to extract fields using regex
        try:
            # Company name - look for patterns like "company": "Name" or "company name": "Name"
            company_match = re.search(r'(?:"client_company"|"company"|"company name")\s*:\s*"([^"]+)"', raw_output)
            if company_match:
                result["client_company"] = company_match.group(1)

            # Email - look for email pattern
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', raw_output)
            if email_match:
                result["client_email"] = email_match.group(0)

            # Order total - look for dollar amounts
            total_match = re.search(r'(?:"order_total"|"total"|"price"|"amount")\s*:\s*(\d+(?:\.\d+)?)', raw_output)
            if total_match:
                result["order_total"] = float(total_match.group(1))

            # Invoice numbers - typically alphanumeric codes
            invoice_match = re.search(r'(?:"sales_invoice"|"invoice")\s*:\s*"([^"]+)"', raw_output)
            if invoice_match:
                result["sales_invoice"] = invoice_match.group(1)

            # Purchase order
            po_match = re.search(r'(?:"purchase_invoice"|"purchase order"|"po")\s*:\s*"([^"]+)"', raw_output)
            if po_match:
                result["purchase_invoice"] = po_match.group(1)

            # Classification - check if any of the expected values appear
            if "quote-related" in raw_output.lower():
                result["classification"] = "Quote-related"
            elif "quote-adjacent" in raw_output.lower():
                result["classification"] = "Quote-adjacent"
            elif "non-quote" in raw_output.lower():
                result["classification"] = "Non-quote adjacent"
        except Exception as ex:
            logger.error(f"Error during regex extraction: {str(ex)}")

        return result


# === PARALLEL PROCESSING ===
def process_batch(batch_items, transcripts):
    """Process a batch of conversations in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future-to-id mapping
        future_to_id = {
            executor.submit(process_single_thread, thread_id, prompt, transcripts[thread_id]): thread_id
            for thread_id, prompt in batch_items
        }

        # Process as completed with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_id),
                           total=len(future_to_id),
                           desc="Processing batch"):
            thread_id = future_to_id[future]
            try:
                result = future.result()
                results[thread_id] = result
                logger.info(f"Thread {thread_id} processed: {result}")
            except Exception as e:
                logger.error(f"Error processing thread {thread_id}: {str(e)}")
                results[thread_id] = {
                    "classification": "Error",
                    "reason": f"Processing error: {str(e)}",
                    "client_company": None,
                    "client_email": None,
                    "order_total": None,
                    "purchase_invoice": None,
                    "sales_invoice": None
                }
    return results


def process_single_thread(thread_id, prompt, transcript):
    """Process a single conversation thread."""
    output = run_gemma3(prompt)
    if output:
        result = postprocess_output(output)
        result['thread_id'] = thread_id
        return result
    else:
        return {
            "classification": "Error processing",
            "reason": "No output from LLM",
            "client_company": None,
            "client_email": None,
            "order_total": None,
            "purchase_invoice": None,
            "sales_invoice": None,
            "thread_id": thread_id
        }


# === FIRST PASS FILTERING ===
def perform_first_pass(transcripts):
    """
    Perform the first pass filtering to identify potential quote-related conversations.
    Returns a dictionary of filtered thread IDs with their confidence scores and reasons.
    """
    print("Starting first pass filtering...")
    logger.info("Starting first pass filtering...")

    first_pass_results = {}
    potentially_relevant = {}

    for thread_id, transcript in tqdm(transcripts.items(), desc="First pass filtering"):
        is_relevant, confidence, reason = first_pass_analysis(transcript)
        first_pass_results[thread_id] = {
            "is_potentially_relevant": is_relevant,
            "confidence": confidence,
            "reason": reason
        }

        if is_relevant:
            potentially_relevant[thread_id] = first_pass_results[thread_id]

    # Save the first pass results
    save_results(first_pass_results, FIRST_PASS_FILE)

    print(f"First pass complete. Identified {len(potentially_relevant)} potentially relevant threads.")
    logger.info(f"First pass complete. Identified {len(potentially_relevant)} potentially relevant threads.")

    return potentially_relevant


# === MAIN EXECUTION ===
def main():
    # Force a new run by deleting output files if FORCE_RESET is True
    if FORCE_RESET:
        for file in [OUTPUT_FILE, FULL_DETAILS_OUTPUT_FILE, FIRST_PASS_FILE, SUMMARY_FILE]:
            if os.path.exists(file):
                os.remove(file)
                logger.info(f"Existing {file} removed for a fresh run.")
                print(f"Existing {file} removed for a fresh run.")

    # Load transcripts
    transcripts = load_transcripts(TRANSCRIPTS_FILE)
    total_threads = len(transcripts)
    print(f"Total conversation threads available: {total_threads}")
    logger.info(f"Total conversation threads available: {total_threads}")

    # === FIRST PASS: Filter potentially relevant threads ===
    if os.path.exists(FIRST_PASS_FILE) and not FORCE_RESET:
        print("Loading existing first pass results...")
        with open(FIRST_PASS_FILE, 'r', encoding='utf-8') as f:
            first_pass_results = json.load(f)
        potentially_relevant = {thread_id: data for thread_id, data in first_pass_results.items()
                                if data.get('is_potentially_relevant', False)}
        print(f"Loaded {len(potentially_relevant)} potentially relevant threads from previous run.")
    else:
        potentially_relevant = perform_first_pass(transcripts)

    # === SECOND PASS: Process potentially relevant threads with LLM ===
    print(f"\nStarting second pass processing with LLM for {len(potentially_relevant)} threads...")
    logger.info(f"Starting second pass processing with LLM for {len(potentially_relevant)} threads...")

    # Initialize results dictionaries
    results = {}
    full_details_results = {}

    # Prepare prompts for LLM processing
    prompt_data = {}
    for thread_id in potentially_relevant:
        prompt_data[thread_id] = build_prompt(thread_id, transcripts[thread_id])["prompt"]

    # Process in batches
    all_items = list(prompt_data.items())
    total_batches = (len(all_items) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Processing {len(all_items)} threads in {total_batches} batches...")
    logger.info(f"Processing {len(all_items)} threads in {total_batches} batches...")

    for batch_index in range(total_batches):
        start_idx = batch_index * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(all_items))
        current_batch = all_items[start_idx:end_idx]

        print(f"Processing batch {batch_index + 1}/{total_batches} ({len(current_batch)} items)")
        logger.info(f"Processing batch {batch_index + 1}/{total_batches} ({len(current_batch)} items)")

        batch_results = process_batch(current_batch, transcripts)
        results.update(batch_results)
        full_details_results.update(batch_results)

        # Save checkpoint after each batch if configured
        if (batch_index + 1) % BATCH_CHECKPOINT_INTERVAL == 0 or batch_index == total_batches - 1:
            save_results(results, OUTPUT_FILE)
            save_results(full_details_results, FULL_DETAILS_OUTPUT_FILE)
            logger.info(f"Batch {batch_index + 1} checkpoint saved.")

    # Generate summary statistics
    print("\nGenerating summary statistics...")

    # Count by classification
    classification_counts = {}
    for thread_id, data in results.items():
        classification = data.get('classification', 'Unknown')
        classification_counts[classification] = classification_counts.get(classification, 0) + 1

    # Count threads with financial information
    threads_with_totals = sum(1 for data in results.values() if data.get('order_total') is not None)
    threads_with_invoices = sum(1 for data in results.values()
                                if data.get('sales_invoice') is not None or data.get('purchase_invoice') is not None)

    # Generate summary
    summary = {
        "first_pass": {
            "total_threads": total_threads,
            "potentially_relevant_threads": len(potentially_relevant),
            "filtering_rate": round((1 - len(potentially_relevant) / total_threads) * 100, 2)
        },
        "second_pass": {
            "processed_threads": len(results),
            "classification_breakdown": classification_counts,
            "threads_with_financials": threads_with_totals,
            "threads_with_invoices": threads_with_invoices
        },
        "processing_stats": {
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "log_file": log_filename
        }
    }

    # Save summary
    save_results(summary, SUMMARY_FILE)

    print("\n🎉 Processing complete!")
    print(f"First pass filtered {summary['first_pass']['filtering_rate']}% of threads")
    print(f"Second pass identified {classification_counts.get('Quote-related', 0)} quote-related threads")
    print(f"Full results saved to {OUTPUT_FILE}")
    print(f"Summary saved to {SUMMARY_FILE}")

    logger.info("Processing completed successfully.")


if __name__ == "__main__":
    main()