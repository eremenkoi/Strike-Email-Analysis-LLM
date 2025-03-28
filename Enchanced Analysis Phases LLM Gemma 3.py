#!/usr/bin/env python3
import json
import os
import re
import logging
import time
from datetime import datetime
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import html
import concurrent.futures

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
EXISTING_RESULTS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\quote_conversations_filtered_enhanced.json"
DETAILED_EXTRACTION_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\detailed_extraction_results.json"
COMPREHENSIVE_ANALYSIS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\comprehensive_conversation_analysis.json"
SUMMARY_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\extraction_summary_report.json"
LLM_MODEL = "gemma3:12b"  # Model name for Ollama
BATCH_SIZE = 5  # Smaller batch size for the more intensive prompts
MAX_WORKERS = 4  # Number of parallel workers
FORCE_RESET = False  # Set to True to redo existing analyses
LLM_TIMEOUT = 600  # Timeout in seconds (5 minutes)
SKIP_DETAILED = True         # Set to True to skip detailed extraction
SKIP_COMPREHENSIVE = False  # Keep this False to run comprehensive analysis

# Set up logging
log_filename = f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Enhanced analysis started.")


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


# === HELPER FUNCTIONS ===
def load_json_file(file_path):
    """Load JSON from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        print(f"Error loading file {file_path}: {str(e)}")
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



# === PROMPT ENGINEERING FOR DETAILED EXTRACTION ===
def build_enhanced_prompt(thread_id, transcript):
    """
    Construct a detailed extraction prompt that captures entities, timeline, quote details,
    and multiple reference numbers including reseller relationships.
    """
    conversation_text = ""
    for msg in transcript:
        body = truncate_text(msg.get('body', ""), max_words=150)  # Allow more text for context
        attachment = truncate_text(msg.get('attachment_text', ""), max_words=150)
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
        "You are an experienced sales analyst at strike.com.au. Extract detailed structured information "
        "from this email conversation about a product quote or purchase. "
        "Return ONLY a JSON object with the following structure:\n\n"

        "{\n"
        "  \"classification\": string (one of \"Quote-related\", \"Quote-adjacent\", \"Non-quote adjacent\"),\n"
        "  \"entities\": [\n"
        "    {\n"
        "      \"entity\": string (name of person or organization),\n"
        "      \"role\": string (job title or function),\n"
        "      \"company\": string (company name),\n"
        "      \"contact_info\": {\n"
        "        \"phone\": string (phone number if available),\n"
        "        \"email\": string (email address if available)\n"
        "      }\n"
        "    },\n"
        "    {\n"
        "      \"entity\": string (product name),\n"
        "      \"type\": \"Product\",\n"
        "      \"description\": string (product description),\n"
        "      \"sku\": string (product SKU if mentioned),\n"
        "      \"price\": string (price if mentioned),\n"
        "      \"quantity\": number (quantity requested)\n"
        "    }\n"
        "  ],\n"
        "  \"timeline\": [\n"
        "    {\n"
        "      \"date\": string (in YYYY-MM-DD format),\n"
        "      \"event\": string (description of what happened)\n"
        "    }\n"
        "  ],\n"
        "  \"quote_details\": {\n"
        "    \"total_amount\": number or null,\n"
        "    \"currency\": string,\n"
        "    \"purchase_order\": string or null,\n"
        "    \"sales_invoice\": string or null,\n"
        "    \"delivery_timeframe\": string or null,\n"
        "    \"special_requirements\": string or null,\n"
        "    \"quote_status\": string (\"requested\", \"submitted\", \"negotiating\", \"accepted\", \"rejected\")\n"
        "  },\n"
        "  \"reference_numbers\": {\n"
        "    \"purchase_order\": string or null,\n"
        "    \"sales_invoice\": string or null,\n"
        "    \"supplier_order_number\": string or null,\n"
        "    \"customer_order_number\": string or null,\n"
        "    \"internal_references\": [string],\n"
        "    \"other_references\": [string]\n"
        "  },\n"
        "  \"reseller_information\": {\n"
        "    \"is_reseller_order\": boolean,\n"
        "    \"reseller_company\": string or null,\n"
        "    \"end_customer\": string or null,\n"
        "    \"notes\": string or null\n"
        "  }\n"
        "}\n\n"

        "IMPORTANT INSTRUCTIONS:\n\n"
        "1. Pay special attention to ALL reference numbers and order IDs mentioned anywhere in the conversation.\n"
        "2. Look for reseller relationships where one company (like JB Hi-Fi) is placing an order on behalf of another company (like Thiess).\n"
        "3. If you see multiple order numbers, capture all of them. Some may be supplier references, some customer references.\n"
        "4. Check for reference numbers embedded in product descriptions or special notes (often in ***reference*** format).\n\n"

        "ANALYZE THIS CONVERSATION:\n"
        f"{conversation_text}\n\n"
        "Return ONLY valid JSON without additional text, markdown, or explanation."
    )
    return prompt


# === PROMPT ENGINEERING FOR COMPREHENSIVE ANALYSIS ===
def build_comprehensive_prompt(thread_id, transcript):
    """
    Construct a comprehensive prompt for detailed conversation analysis including
    sales markers, communication patterns, and sentiment analysis.
    """
    # Determine if the conversation is inbound or outbound
    first_sender = transcript[0].get('sender', '') if transcript else ''
    direction = "outbound" if "strike.com" in first_sender.lower() else "inbound"

    # Format full transcript for analysis
    conversation_text = ""
    for msg in transcript:
        body = truncate_text(msg.get('body', ""), max_words=300)  # Allow more text
        attachment = truncate_text(msg.get('attachment_text', ""), max_words=150)
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
            "You are an expert sales and communication analyst at strike.com.au. Analyze this email conversation "
            "and extract detailed information about the communication patterns, sales markers, and "
            "overall effectiveness. Return ONLY a JSON object following this structure:\n\n"

            "{\n"
            "  \"conversation_id\": \"" + thread_id + "\",\n"
                                                      "  \"conversation_direction\": \"" + direction + "\",\n"
                                                                                                       "  \"initiator_email\": string (email of person who started the conversation),\n"
                                                                                                       "  \"total_email_count\": integer,\n"
                                                                                                       "  \"emails\": [\n"
                                                                                                       "    {\n"
                                                                                                       "      \"email_id\": integer (position in conversation),\n"
                                                                                                       "      \"date_time\": string (date from the email),\n"
                                                                                                       "      \"sender\": {\n"
                                                                                                       "        \"name\": string,\n"
                                                                                                       "        \"email\": string,\n"
                                                                                                       "        \"role\": string (buyer/seller/other),\n"
                                                                                                       "        \"seniority_or_role\": string (job title if available)\n"
                                                                                                       "      },\n"
                                                                                                       "      \"recipient\": {\n"
                                                                                                       "        \"name\": string,\n"
                                                                                                       "        \"email\": string,\n"
                                                                                                       "        \"role\": string (buyer/seller/other),\n"
                                                                                                       "        \"seniority_or_role\": string\n"
                                                                                                       "      },\n"
                                                                                                       "      \"response_time_hours_since_prev_email\": number or null,\n"
                                                                                                       "      \"email_analysis\": {\n"
                                                                                                       "        \"sentiment\": string (positive/neutral/negative),\n"
                                                                                                       "        \"urgency_or_scarcity_used\": boolean,\n"
                                                                                                       "        \"personalization\": boolean,\n"
                                                                                                       "        \"language_formality\": string (formal/casual/mixed),\n"
                                                                                                       "        \"email_length_words\": integer,\n"
                                                                                                       "        \"use_of_emojis\": boolean,\n"
                                                                                                       "        \"attachment_or_visual_included\": boolean,\n"
                                                                                                       "        \"questions_asked\": boolean,\n"
                                                                                                       "        \"questions_count\": integer,\n"
                                                                                                       "        \"clear_cta_present\": boolean,\n"
                                                                                                       "        \"exclamation_marks_count\": integer,\n"
                                                                                                       "        \"all_caps_words_used\": boolean,\n"
                                                                                                       "        \"formatting_used\": boolean\n"
                                                                                                       "      },\n"
                                                                                                       "      \"offer_or_request_details\": {\n"
                                                                                                       "        \"product_or_service\": string or null,\n"
                                                                                                       "        \"quantity\": integer or null,\n"
                                                                                                       "        \"pricing_information_provided\": boolean,\n"
                                                                                                       "        \"discount_or_special_offer\": boolean,\n"
                                                                                                       "        \"availability_status\": string,\n"
                                                                                                       "        \"specific_eta_or_timeline_provided\": boolean\n"
                                                                                                       "      },\n"
                                                                                                       "      \"engagement_and_decision\": {\n"
                                                                                                       "        \"decision_maker_involved\": boolean,\n"
                                                                                                       "        \"multiple_stakeholders_ccd\": boolean,\n"
                                                                                                       "        \"objections_raised\": boolean,\n"
                                                                                                       "        \"objection_type\": string or null,\n"
                                                                                                       "        \"next_steps_mentioned\": boolean,\n"
                                                                                                       "        \"competitor_mentioned\": boolean\n"
                                                                                                       "      }\n"
                                                                                                       "    }\n"
                                                                                                       "  ],\n"
                                                                                                       "  \"conversation_analysis\": {\n"
                                                                                                       "    \"overall_sentiment\": string (positive/neutral/negative),\n"
                                                                                                       "    \"sales_cycle_stage\": string (inquiry/evaluation/negotiation/decision/post-purchase),\n"
                                                                                                       "    \"major_pain_points_identified\": [string],\n"
                                                                                                       "    \"key_decision_factors\": [string],\n"
                                                                                                       "    \"competitive_positioning\": string or null,\n"
                                                                                                       "    \"outcome\": string (won/lost/ongoing/unknown),\n"
                                                                                                       "    \"average_response_time_hours\": number,\n"
                                                                                                       "    \"notable_patterns\": [string]\n"
                                                                                                       "  }\n"
                                                                                                       "}\n\n"

                                                                                                       "ANALYZE THIS CONVERSATION:\n"
                                                                                                       f"{conversation_text}\n\n"
                                                                                                       "Return ONLY valid JSON without additional text, markdown, or explanation."
    )
    return prompt


# === LLM INTERACTION ===
def run_gemma3_with_timeout(prompt_text, timeout=LLM_TIMEOUT):
    """
    Run the LLM model using either direct API or subprocess with a timeout.
    """
    start_time = time.time()

    # Define the function to run within the executor
    def run_model():
        try:
            if USING_CLIENT:
                # Use Ollama Python client
                response = ollama_client.chat(model=LLM_MODEL, messages=[
                    {"role": "user", "content": prompt_text}
                ])
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
                return result.stdout
        except Exception as e:
            logger.error(f"Error running LLM: {str(e)}")
            print(f"Error running LLM: {str(e)}")
            return None

    # Run with timeout
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the task and get a future
            future = executor.submit(run_model)

            try:
                # Wait for the result with a timeout
                result = future.result(timeout=timeout)
                duration = time.time() - start_time
                logger.info(f"LLM query completed in {duration:.2f} seconds")
                return result
            except concurrent.futures.TimeoutError:
                logger.warning(f"LLM query timed out after {timeout} seconds")
                print(f"⚠️ Query timed out after {timeout} seconds")
                return None
    except Exception as e:
        logger.error(f"Error in run_gemma3_with_timeout: {str(e)}")
        return None


def extract_json_safely(raw_output, error_template):
    """
    Generic function to extract JSON from LLM output with error handling.
    """
    if not raw_output:
        return error_template

    # First try to extract JSON if it's embedded in other text
    json_pattern = r'(\{.*\})'
    json_matches = re.findall(json_pattern, raw_output, re.DOTALL)

    if json_matches:
        # Use the largest JSON object found (most complete)
        candidate = max(json_matches, key=len)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # If this fails, fall back to the full text parsing

    # Try to parse the full output as JSON
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e} | Raw output: {raw_output.strip()}")
        error_template["error"] = f"Failed to decode JSON: {str(e)}"
        error_template["raw_output"] = raw_output.strip()
        return error_template


# === EXTRACTION FUNCTIONS ===
def perform_detailed_extraction(thread_id, transcript):
    """
    For threads already identified as quote-related, perform a more detailed extraction
    to get entities, timeline, and structured quote details.
    """
    prompt = build_enhanced_prompt(thread_id, transcript)
    output = run_gemma3_with_timeout(prompt, timeout=LLM_TIMEOUT)

    if output is None:
        # Query timed out
        return {
            "classification": "Error",
            "reason": f"LLM query timed out after {LLM_TIMEOUT} seconds",
            "entities": [],
            "timeline": [],
            "quote_details": {
                "total_amount": None,
                "currency": None,
                "purchase_order": None,
                "sales_invoice": None,
                "delivery_timeframe": None,
                "special_requirements": None,
                "quote_status": None
            }
        }

    error_template = {
        "classification": "Error",
        "reason": "No output from LLM",
        "entities": [],
        "timeline": [],
        "quote_details": {
            "total_amount": None,
            "currency": None,
            "purchase_order": None,
            "sales_invoice": None,
            "delivery_timeframe": None,
            "special_requirements": None,
            "quote_status": None
        }
    }

    return extract_json_safely(output, error_template)


def perform_comprehensive_analysis(thread_id, transcript):
    """
    Perform comprehensive conversation analysis including communication patterns,
    sales markers, and sentiment.
    """
    prompt = build_comprehensive_prompt(thread_id, transcript)
    output = run_gemma3_with_timeout(prompt, timeout=LLM_TIMEOUT)

    error_template = {
        "conversation_id": thread_id,
        "error": "Failed to analyze conversation",
        "emails": [],
        "conversation_analysis": {
            "overall_sentiment": "unknown",
            "sales_cycle_stage": "unknown",
            "outcome": "unknown"
        }
    }

    return extract_json_safely(output, error_template)


# === ANALYSIS FUNCTIONS ===
def find_unprocessed_threads(processed_ids, all_threads, filter_func=None):
    """
    Find thread IDs that haven't been processed yet.

    Args:
        processed_ids: Set of thread IDs that have already been processed
        all_threads: Dictionary of all threads
        filter_func: Optional function to filter threads (e.g., only quote-related)

    Returns:
        List of thread IDs that need processing
    """
    if filter_func is None:
        # Default filter: get quote-related threads
        filter_func = lambda data: data.get("classification") == "Quote-related"

    unprocessed = [
        thread_id for thread_id, data in all_threads.items()
        if filter_func(data) and thread_id not in processed_ids
    ]

    return unprocessed


# === PARALLEL PROCESSING ===
def process_batch(batch_threads, transcripts, processor_func):
    """Process a batch of threads using the specified processor function."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(processor_func, thread_id, transcripts[thread_id]): thread_id
            for thread_id in batch_threads
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_id),
                           total=len(future_to_id),
                           desc=processor_func.__name__):
            thread_id = future_to_id[future]
            try:
                result = future.result()
                results[thread_id] = result
                logger.info(f"{processor_func.__name__} for thread {thread_id} completed")
            except Exception as e:
                logger.error(f"Error in {processor_func.__name__} for {thread_id}: {str(e)}")
                # Create a generic error result based on the processor type
                if processor_func.__name__ == "perform_detailed_extraction":
                    results[thread_id] = {
                        "error": f"Processing error: {str(e)}",
                        "entities": [],
                        "timeline": [],
                        "quote_details": None
                    }
                else:
                    results[thread_id] = {
                        "conversation_id": thread_id,
                        "error": f"Processing error: {str(e)}",
                        "emails": [],
                        "conversation_analysis": {"outcome": "unknown"}
                    }

    return results


# === MAIN EXECUTION ===
def main():
    # Print banner
    print("\n==== ENHANCED QUOTE ANALYSIS SYSTEM ====")
    print("Adding detailed entity extraction and conversation analysis to existing results\n")

    # Load the existing classification results
    print("Loading existing classification results...")
    existing_results = load_json_file(EXISTING_RESULTS_FILE)

    if not existing_results:
        print("❌ Error: No existing classification results found. Exiting.")
        return

    print(f"Loaded {len(existing_results)} classified threads.")

    # Load the full conversation transcripts
    print("Loading conversation transcripts...")
    transcripts = load_json_file(TRANSCRIPTS_FILE)

    if not transcripts:
        print("❌ Error: No conversation transcripts found. Exiting.")
        return

    print(f"Loaded {len(transcripts)} conversation transcripts.")

    # Filter for quote-related threads
    quote_related_threads = [
        thread_id for thread_id, data in existing_results.items()
        if data.get("classification") == "Quote-related"
    ]

    print(f"Found {len(quote_related_threads)} quote-related threads for detailed analysis.")

    # === THIRD PASS: Detailed entity and timeline extraction ===
    if not SKIP_DETAILED:
        detailed_results = {}

        if os.path.exists(DETAILED_EXTRACTION_FILE) and not FORCE_RESET:
            print("Loading existing detailed extraction results...")
            detailed_results = load_json_file(DETAILED_EXTRACTION_FILE)
            print(f"Loaded detailed extraction for {len(detailed_results)} threads.")

            # Find threads that still need processing
            processed_detailed_ids = set(detailed_results.keys())
            remaining_detailed_threads = find_unprocessed_threads(processed_detailed_ids, existing_results)

            if remaining_detailed_threads:
                print(
                    f"\nResuming detailed entity extraction for {len(remaining_detailed_threads)} remaining threads...")
                logger.info(
                    f"Resuming detailed entity extraction for {len(remaining_detailed_threads)} remaining threads.")

                # Process remaining threads in batches
                for i in range(0, len(remaining_detailed_threads), BATCH_SIZE):
                    batch = remaining_detailed_threads[i:i + BATCH_SIZE]
                    print(
                        f"Processing batch {i // BATCH_SIZE + 1}/{(len(remaining_detailed_threads) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} threads)")

                    batch_results = process_batch(batch, transcripts, perform_detailed_extraction)
                    detailed_results.update(batch_results)

                    # Save checkpoint after each batch
                    save_results(detailed_results, DETAILED_EXTRACTION_FILE)
                    logger.info(f"Saved checkpoint after batch {i // BATCH_SIZE + 1}")

                print(f"Detailed entity extraction complete. Results saved to {DETAILED_EXTRACTION_FILE}")
            else:
                print("✓ All threads have already been processed for detailed extraction.")
        else:
            print("\nStarting detailed entity extraction...")
            logger.info("Starting detailed entity extraction for quote-related threads.")

            # Process all threads in batches
            for i in range(0, len(quote_related_threads), BATCH_SIZE):
                batch = quote_related_threads[i:i + BATCH_SIZE]
                print(
                    f"Processing batch {i // BATCH_SIZE + 1}/{(len(quote_related_threads) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} threads)")

                batch_results = process_batch(batch, transcripts, perform_detailed_extraction)
                detailed_results.update(batch_results)

                # Save checkpoint after each batch
                save_results(detailed_results, DETAILED_EXTRACTION_FILE)
                logger.info(f"Saved checkpoint after batch {i // BATCH_SIZE + 1}")

            print(f"Detailed entity extraction complete. Results saved to {DETAILED_EXTRACTION_FILE}")
    else:
        print("⚠️ Skipping detailed entity extraction as requested.")

    # === FOURTH PASS: Comprehensive conversation analysis ===
    if not SKIP_COMPREHENSIVE:
        comprehensive_results = {}

        if os.path.exists(COMPREHENSIVE_ANALYSIS_FILE) and not FORCE_RESET:
            print("Loading existing comprehensive analysis results...")
            comprehensive_results = load_json_file(COMPREHENSIVE_ANALYSIS_FILE)
            print(f"Loaded comprehensive analysis for {len(comprehensive_results)} threads.")

            # Find threads that still need processing
            processed_comp_ids = set(comprehensive_results.keys())
            remaining_comp_threads = find_unprocessed_threads(processed_comp_ids, existing_results)

            if remaining_comp_threads:
                print(
                    f"\nResuming comprehensive conversation analysis for {len(remaining_comp_threads)} remaining threads...")
                logger.info(
                    f"Resuming comprehensive conversation analysis for {len(remaining_comp_threads)} remaining threads.")

                # Process remaining threads in batches
                for i in range(0, len(remaining_comp_threads), BATCH_SIZE):
                    batch = remaining_comp_threads[i:i + BATCH_SIZE]
                    print(
                        f"Processing batch {i // BATCH_SIZE + 1}/{(len(remaining_comp_threads) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} threads)")

                    batch_results = process_batch(batch, transcripts, perform_comprehensive_analysis)
                    comprehensive_results.update(batch_results)

                    # Save checkpoint after each batch
                    save_results(comprehensive_results, COMPREHENSIVE_ANALYSIS_FILE)
                    logger.info(f"Saved checkpoint after batch {i // BATCH_SIZE + 1}")

                print(f"Comprehensive conversation analysis complete. Results saved to {COMPREHENSIVE_ANALYSIS_FILE}")
            else:
                print("✓ All threads have already been processed for comprehensive analysis.")
        else:
            print("\nStarting comprehensive conversation analysis...")
            logger.info("Starting comprehensive conversation analysis.")

            # Process all threads in batches
            for i in range(0, len(quote_related_threads), BATCH_SIZE):
                batch = quote_related_threads[i:i + BATCH_SIZE]
                print(
                    f"Processing batch {i // BATCH_SIZE + 1}/{(len(quote_related_threads) + BATCH_SIZE - 1) // BATCH_SIZE} ({len(batch)} threads)")

                batch_results = process_batch(batch, transcripts, perform_comprehensive_analysis)
                comprehensive_results.update(batch_results)

                # Save checkpoint after each batch
                save_results(comprehensive_results, COMPREHENSIVE_ANALYSIS_FILE)
                logger.info(f"Saved checkpoint after batch {i // BATCH_SIZE + 1}")

            print(f"Comprehensive conversation analysis complete. Results saved to {COMPREHENSIVE_ANALYSIS_FILE}")
    else:
        print("⚠️ Skipping comprehensive conversation analysis as requested.")

    # Generate summary statistics if we have results
    if (not SKIP_DETAILED and detailed_results) or (not SKIP_COMPREHENSIVE and comprehensive_results):
        print("\nGenerating summary statistics...")

        # Get the results to analyze
        if not SKIP_DETAILED:
            detailed_results = load_json_file(DETAILED_EXTRACTION_FILE)
        else:
            detailed_results = {}

        if not SKIP_COMPREHENSIVE:
            comprehensive_results = load_json_file(COMPREHENSIVE_ANALYSIS_FILE)
        else:
            comprehensive_results = {}

        # Count threads with detailed information
        detailed_success = sum(1 for data in detailed_results.values()
                               if 'error' not in data and len(data.get('entities', [])) > 0)

        # Analyze reseller relationships
        reseller_count = sum(1 for data in detailed_results.values()
                             if data.get('reseller_information', {}).get('is_reseller_order', False))

        # Analyze sales outcomes from comprehensive results
        sales_outcomes = {}
        for thread_id, data in comprehensive_results.items():
            if 'conversation_analysis' in data:
                outcome = data['conversation_analysis'].get('outcome', 'unknown')
                sales_outcomes[outcome] = sales_outcomes.get(outcome, 0) + 1

        # Calculate response time statistics
        response_times = []
        for data in comprehensive_results.values():
            for email in data.get('emails', []):
                response_time = email.get('response_time_hours_since_prev_email')
                if response_time and response_time > 0:
                    response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Generate summary
        summary = {
            "classification_summary": {
                "total_threads": len(existing_results),
                "quote_related_threads": len(quote_related_threads),
                "quote_related_percentage": round(len(quote_related_threads) / len(existing_results) * 100,
                                                  2) if existing_results else 0
            },
            "detailed_extraction_summary": {
                "processed_threads": len(detailed_results),
                "successful_extractions": detailed_success,
                "success_rate": round((detailed_success / len(detailed_results) * 100), 2) if detailed_results else 0,
                "reseller_orders": reseller_count
            },
            "comprehensive_analysis_summary": {
                "processed_threads": len(comprehensive_results),
                "sales_outcomes": sales_outcomes,
                "average_response_time_hours": round(avg_response_time, 2)
            },
            "processing_stats": {
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "log_file": log_filename
            }
        }

        # Save summary
        save_results(summary, SUMMARY_FILE)

        print("\n🎉 Enhanced analysis complete!")
        print(f"Processed {len(quote_related_threads)} quote-related threads")
        if not SKIP_DETAILED:
            print(f"Detailed entity extraction successful for {detailed_success} threads")
            print(f"Identified {reseller_count} reseller orders")
        if not SKIP_COMPREHENSIVE:
            print(f"Sales outcomes identified: {sales_outcomes}")
            print(f"Average response time: {round(avg_response_time, 2)} hours")

        print(f"\nFull results saved to:")
        if not SKIP_DETAILED:
            print(f"- Detailed entity extraction: {DETAILED_EXTRACTION_FILE}")
        if not SKIP_COMPREHENSIVE:
            print(f"- Comprehensive conversation analysis: {COMPREHENSIVE_ANALYSIS_FILE}")
        print(f"- Summary report: {SUMMARY_FILE}")

    logger.info("Enhanced analysis completed successfully.")


if __name__ == "__main__":
    main()