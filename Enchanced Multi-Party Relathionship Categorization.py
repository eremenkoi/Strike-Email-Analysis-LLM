#!/usr/bin/env python3
import json
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import re
from fuzzywuzzy import fuzz, process

# === CONFIGURATION ===
DETAILED_EXTRACTION_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\detailed_extraction_results.json"
COMPREHENSIVE_ANALYSIS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\comprehensive_conversation_analysis.json"
CLIENT_CATEGORIZATION_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\client_categorization_results.json"
PRODUCT_CATEGORIZATION_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\product_categorization_results.json"
RESELLER_RELATIONSHIPS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\reseller_relationships_results.json"
COMBINED_ANALYSIS_FILE = r"C:\Users\ierem\PycharmProjects\pythonProject\combined_analysis_results.json"
VISUALIZATION_FOLDER = r"C:\Users\ierem\PycharmProjects\pythonProject\visualizations"

# Ensure visualization folder exists
os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)

# Set up logging
log_filename = f"categorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
logger.info("Enhanced categorization processing started.")


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


def analyze_communication_effectiveness(comprehensive_results):
    """
    Analyze the effectiveness of different communication styles and patterns.
    Returns insights on what communication approaches correlate with better outcomes.
    """
    print("Analyzing communication effectiveness...")
    logger.info("Starting communication effectiveness analysis.")

    # Initialize results structure
    communication_insights = {
        'email_length': {
            'bins': {
                'short': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
                'medium': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
                'long': {'count': 0, 'success_rate': 0, 'total_conversions': 0}
            },
            'correlation': None
        },
        'formality': {
            'formal': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'informal': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'correlation': None
        },
        'personalization': {
            'personalized': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'non_personalized': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'correlation': None
        },
        'cta_effectiveness': {
            'clear_cta': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'no_clear_cta': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'correlation': None
        },
        'question_usage': {
            'with_questions': {'count': 0, 'success_rate': 0, 'avg_questions': 0, 'total_conversions': 0},
            'no_questions': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'correlation': None
        },
        'response_time': {
            'quick': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'medium': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'slow': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'correlation': None
        },
        'visual_content': {
            'with_visuals': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'no_visuals': {'count': 0, 'success_rate': 0, 'total_conversions': 0},
            'correlation': None
        },
        'top_patterns': [],
        'winning_combination': None
    }

    # Track raw data for correlation analysis
    email_lengths = []
    question_counts = []
    conversion_results = []  # 1 for conversion, 0 for no conversion

    # Process each conversation
    for conv_id, conv_data in comprehensive_results.items():
        # Skip conversations with no emails
        if 'emails' not in conv_data or not conv_data['emails']:
            continue

        # Determine if the conversation resulted in a conversion/sale
        outcome = conv_data.get('conversation_analysis', {}).get('outcome', '').lower()
        sales_cycle = conv_data.get('conversation_analysis', {}).get('sales_cycle_stage', '').lower()

        # Define what counts as a successful conversion
        is_conversion = outcome in ['won', 'closed', 'successful'] or sales_cycle in ['closed won', 'post-purchase']
        conversion_results.append(1 if is_conversion else 0)

        # Get initial email attributes (from first email in conversation)
        first_email = conv_data['emails'][0]
        email_analysis = first_email.get('email_analysis', {})

        # Email length analysis
        word_count = email_analysis.get('email_length_words', 0)
        email_lengths.append(word_count)

        if word_count < 100:
            communication_insights['email_length']['bins']['short']['count'] += 1
            if is_conversion:
                communication_insights['email_length']['bins']['short']['total_conversions'] += 1
        elif word_count < 200:
            communication_insights['email_length']['bins']['medium']['count'] += 1
            if is_conversion:
                communication_insights['email_length']['bins']['medium']['total_conversions'] += 1
        else:
            communication_insights['email_length']['bins']['long']['count'] += 1
            if is_conversion:
                communication_insights['email_length']['bins']['long']['total_conversions'] += 1

        # Formality analysis
        formality = email_analysis.get('language_formality', 'formal').lower()
        if formality == 'formal':
            communication_insights['formality']['formal']['count'] += 1
            if is_conversion:
                communication_insights['formality']['formal']['total_conversions'] += 1
        else:
            communication_insights['formality']['informal']['count'] += 1
            if is_conversion:
                communication_insights['formality']['informal']['total_conversions'] += 1

        # Personalization analysis
        is_personalized = email_analysis.get('personalization', False)
        if is_personalized:
            communication_insights['personalization']['personalized']['count'] += 1
            if is_conversion:
                communication_insights['personalization']['personalized']['total_conversions'] += 1
        else:
            communication_insights['personalization']['non_personalized']['count'] += 1
            if is_conversion:
                communication_insights['personalization']['non_personalized']['total_conversions'] += 1

        # CTA analysis
        has_cta = email_analysis.get('clear_cta_present', False)
        if has_cta:
            communication_insights['cta_effectiveness']['clear_cta']['count'] += 1
            if is_conversion:
                communication_insights['cta_effectiveness']['clear_cta']['total_conversions'] += 1
        else:
            communication_insights['cta_effectiveness']['no_clear_cta']['count'] += 1
            if is_conversion:
                communication_insights['cta_effectiveness']['no_clear_cta']['total_conversions'] += 1

        # Question usage analysis
        has_questions = email_analysis.get('questions_asked', False)
        question_count = email_analysis.get('questions_count', 0)
        question_counts.append(question_count)

        if has_questions:
            communication_insights['question_usage']['with_questions']['count'] += 1
            communication_insights['question_usage']['with_questions']['avg_questions'] += question_count
            if is_conversion:
                communication_insights['question_usage']['with_questions']['total_conversions'] += 1
        else:
            communication_insights['question_usage']['no_questions']['count'] += 1
            if is_conversion:
                communication_insights['question_usage']['no_questions']['total_conversions'] += 1

        # Visual content analysis
        has_visuals = email_analysis.get('attachment_or_visual_included', False)
        if has_visuals:
            communication_insights['visual_content']['with_visuals']['count'] += 1
            if is_conversion:
                communication_insights['visual_content']['with_visuals']['total_conversions'] += 1
        else:
            communication_insights['visual_content']['no_visuals']['count'] += 1
            if is_conversion:
                communication_insights['visual_content']['no_visuals']['total_conversions'] += 1

    # Calculate success rates for each category
    # Email length
    for bin_key in communication_insights['email_length']['bins']:
        bin_data = communication_insights['email_length']['bins'][bin_key]
        if bin_data['count'] > 0:
            bin_data['success_rate'] = (bin_data['total_conversions'] / bin_data['count']) * 100

    # Formality
    for key in ['formal', 'informal']:
        if communication_insights['formality'][key]['count'] > 0:
            communication_insights['formality'][key]['success_rate'] = (
                                                                               communication_insights['formality'][key][
                                                                                   'total_conversions'] /
                                                                               communication_insights['formality'][key][
                                                                                   'count']
                                                                       ) * 100

    # Personalization
    for key in ['personalized', 'non_personalized']:
        if communication_insights['personalization'][key]['count'] > 0:
            communication_insights['personalization'][key]['success_rate'] = (
                                                                                     communication_insights[
                                                                                         'personalization'][key][
                                                                                         'total_conversions'] /
                                                                                     communication_insights[
                                                                                         'personalization'][key][
                                                                                         'count']
                                                                             ) * 100

    # CTA
    for key in ['clear_cta', 'no_clear_cta']:
        if communication_insights['cta_effectiveness'][key]['count'] > 0:
            communication_insights['cta_effectiveness'][key]['success_rate'] = (
                                                                                       communication_insights[
                                                                                           'cta_effectiveness'][key][
                                                                                           'total_conversions'] /
                                                                                       communication_insights[
                                                                                           'cta_effectiveness'][key][
                                                                                           'count']
                                                                               ) * 100

    # Questions
    for key in ['with_questions', 'no_questions']:
        if communication_insights['question_usage'][key]['count'] > 0:
            communication_insights['question_usage'][key]['success_rate'] = (
                                                                                    communication_insights[
                                                                                        'question_usage'][key][
                                                                                        'total_conversions'] /
                                                                                    communication_insights[
                                                                                        'question_usage'][key]['count']
                                                                            ) * 100

    # Calculate average questions per email for question-containing emails
    if communication_insights['question_usage']['with_questions']['count'] > 0:
        communication_insights['question_usage']['with_questions']['avg_questions'] /= \
        communication_insights['question_usage']['with_questions']['count']

    # Visual content
    for key in ['with_visuals', 'no_visuals']:
        if communication_insights['visual_content'][key]['count'] > 0:
            communication_insights['visual_content'][key]['success_rate'] = (
                                                                                    communication_insights[
                                                                                        'visual_content'][key][
                                                                                        'total_conversions'] /
                                                                                    communication_insights[
                                                                                        'visual_content'][key]['count']
                                                                            ) * 100

    # Calculate correlations using numpy if available
    try:
        import numpy as np

        # Email length correlation with conversion
        if email_lengths and len(email_lengths) > 2 and len(set(conversion_results)) > 1:
            correlation = np.corrcoef(email_lengths, conversion_results)[0, 1]
            communication_insights['email_length']['correlation'] = correlation

        # Question count correlation with conversion
        if question_counts and len(question_counts) > 2 and len(set(conversion_results)) > 1:
            correlation = np.corrcoef(question_counts, conversion_results)[0, 1]
            communication_insights['question_usage']['correlation'] = correlation
    except:
        # If numpy is not available, skip correlation analysis
        pass

    # Generate insights
    insights = []

    # Email length insight
    highest_success_rate = 0
    best_length = None
    for length, data in communication_insights['email_length']['bins'].items():
        if data['count'] >= 5 and data['success_rate'] > highest_success_rate:
            highest_success_rate = data['success_rate']
            best_length = length

    if best_length:
        insights.append(
            f"{best_length.capitalize()} emails ({communication_insights['email_length']['bins'][best_length]['success_rate']:.1f}% success rate) perform best.")

    # Formality insight
    if communication_insights['formality']['formal']['count'] >= 5 and communication_insights['formality']['informal'][
        'count'] >= 5:
        formal_rate = communication_insights['formality']['formal']['success_rate']
        informal_rate = communication_insights['formality']['informal']['success_rate']
        better_style = "Formal" if formal_rate > informal_rate else "Informal"
        insights.append(
            f"{better_style} communication style has a higher success rate ({max(formal_rate, informal_rate):.1f}%).")

    # Personalization insight
    if communication_insights['personalization']['personalized']['count'] >= 5:
        personalized_rate = communication_insights['personalization']['personalized']['success_rate']
        non_personalized_rate = communication_insights['personalization']['non_personalized']['success_rate']
        if abs(personalized_rate - non_personalized_rate) > 5:  # Only if there's a meaningful difference
            better_approach = "Personalized" if personalized_rate > non_personalized_rate else "Non-personalized"
            insights.append(
                f"{better_approach} emails perform better with a {max(personalized_rate, non_personalized_rate):.1f}% success rate.")

    # CTA insight
    if communication_insights['cta_effectiveness']['clear_cta']['count'] >= 5:
        cta_rate = communication_insights['cta_effectiveness']['clear_cta']['success_rate']
        no_cta_rate = communication_insights['cta_effectiveness']['no_clear_cta']['success_rate']
        if abs(cta_rate - no_cta_rate) > 5:  # Only if there's a meaningful difference
            better_approach = "Emails with clear CTAs" if cta_rate > no_cta_rate else "Emails without explicit CTAs"
            insights.append(
                f"{better_approach} convert better ({max(cta_rate, no_cta_rate):.1f}% vs {min(cta_rate, no_cta_rate):.1f}%).")

    # Question usage insight
    if communication_insights['question_usage']['with_questions']['count'] >= 5:
        with_q_rate = communication_insights['question_usage']['with_questions']['success_rate']
        no_q_rate = communication_insights['question_usage']['no_questions']['success_rate']
        if abs(with_q_rate - no_q_rate) > 5:  # Only if there's a meaningful difference
            better_approach = "Asking questions" if with_q_rate > no_q_rate else "Not asking questions"
            avg_q = communication_insights['question_usage']['with_questions']['avg_questions']
            if with_q_rate > no_q_rate:
                insights.append(
                    f"Asking questions improves success rate by {(with_q_rate - no_q_rate):.1f}%. Best emails average {avg_q:.1f} questions.")
            else:
                insights.append(f"Direct communication without questions performs better in this dataset.")

    # Visual content insight
    if communication_insights['visual_content']['with_visuals']['count'] >= 5:
        visual_rate = communication_insights['visual_content']['with_visuals']['success_rate']
        no_visual_rate = communication_insights['visual_content']['no_visuals']['success_rate']
        if abs(visual_rate - no_visual_rate) > 5:  # Only if there's a meaningful difference
            better_approach = "Including visuals or attachments" if visual_rate > no_visual_rate else "Plain text emails"
            insights.append(
                f"{better_approach} yield higher conversion rates ({max(visual_rate, no_visual_rate):.1f}%).")

    # Add insights to results
    communication_insights['top_patterns'] = insights

    # Determine winning combination
    winning_combo = []

    # Length
    if best_length:
        winning_combo.append(f"{best_length} length")

    # Formality
    if communication_insights['formality']['formal']['success_rate'] > communication_insights['formality']['informal'][
        'success_rate']:
        winning_combo.append("formal style")
    else:
        winning_combo.append("informal style")

    # Questions
    if communication_insights['question_usage']['with_questions']['success_rate'] > \
            communication_insights['question_usage']['no_questions']['success_rate']:
        winning_combo.append("with questions")

    # CTA
    if communication_insights['cta_effectiveness']['clear_cta']['success_rate'] > \
            communication_insights['cta_effectiveness']['no_clear_cta']['success_rate']:
        winning_combo.append("clear CTA")

    if winning_combo:
        communication_insights['winning_combination'] = "Emails that are " + ", ".join(winning_combo)

    print("Communication effectiveness analysis complete.")
    logger.info("Communication effectiveness analysis complete.")

    return communication_insights

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


def visualize_communication_effectiveness(communication_insights):
    """Create visualizations for communication effectiveness insights."""

    # Bar chart comparing success rates of different email lengths
    lengths = communication_insights['email_length']['bins'].keys()
    success_rates = [communication_insights['email_length']['bins'][length]['success_rate'] for length in lengths]

    plt.figure(figsize=(10, 6))
    plt.bar(lengths, success_rates)
    plt.title('Success Rates by Email Length')
    plt.ylabel('Success Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'email_length_success.png'))
    plt.close()

    # Create a comparison chart for all the binary factors
    binary_factors = [
        ('Formality', 'formal', 'informal'),
        ('Personalization', 'personalized', 'non_personalized'),
        ('CTA', 'clear_cta', 'no_clear_cta'),
        ('Questions', 'with_questions', 'no_questions'),
        ('Visuals', 'with_visuals', 'no_visuals')
    ]

    labels = []
    positive_rates = []
    negative_rates = []

    for factor_name, positive_key, negative_key in binary_factors:
        factor_data = communication_insights[factor_name.lower()]
        if factor_data[positive_key]['count'] > 0 and factor_data[negative_key]['count'] > 0:
            labels.append(factor_name)
            positive_rates.append(factor_data[positive_key]['success_rate'])
            negative_rates.append(factor_data[negative_key]['success_rate'])

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], positive_rates, width, label='With Feature')
    plt.bar([i + width / 2 for i in x], negative_rates, width, label='Without Feature')

    plt.ylabel('Success Rate (%)')
    plt.title('Impact of Email Features on Success Rate')
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'email_features_impact.png'))
    plt.close()


def extract_domain(email):
    """Extract domain from email address."""
    if not email or '@' not in email:
        return None
    try:
        return email.split('@')[1].lower()
    except:
        return None

def safe_to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def normalize_company_name(name):
    """Normalize company name for better matching."""
    if not name:
        return None

    # Remove common legal entity suffixes
    suffixes = [' ltd', ' limited', ' inc', ' incorporated', ' llc', ' pty', ' pty ltd', ' co', ' corporation', ' corp',
                ' group', ' australia', ' holdings']
    normalized = name.lower()
    for suffix in suffixes:
        normalized = normalized.replace(suffix, '')

    # Remove spaces, punctuation, and common words
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = normalized.replace('the', '').replace('and', '')
    normalized = ' '.join(normalized.split())  # Normalize whitespace

    return normalized


def safe_extract_total_amount(data):
    """
    Safely extract total amount from different possible locations in the data.

    Args:
        data (dict): The conversation data dictionary

    Returns:
        float: Extracted total amount or 0
    """
    # If data is None, return 0
    if data is None:
        return 0

    # Try extracting from quote_details
    quote_details = data.get('quote_details')

    # If quote_details is None, return 0
    if quote_details is None:
        return 0

    # Try different ways to extract total amount
    amount_keys = ['total_amount', 'total', 'amount', 'value']

    for key in amount_keys:
        try:
            amount = quote_details.get(key)

            # If amount is None, continue to next key
            if amount is None:
                continue

            # Try to convert to float
            if isinstance(amount, str):
                # Remove currency symbols and commas
                amount = amount.replace('$', '').replace(',', '')

            # Convert to float
            return float(amount)
        except (ValueError, TypeError):
            continue

    # If no amount found, return 0
    return 0


def safe_extract_order_details(data, relationship):
    """
    Safely extract order details from conversation data.

    Args:
        data (dict): The conversation data dictionary
        relationship (dict): Reseller relationship information

    Returns:
        dict: Extracted order details
    """
    # Get safe total amount
    total_amount = safe_extract_total_amount(data)

    # Extract order numbers
    order_numbers = extract_order_numbers(data)

    # Find order date from timeline
    order_date = None
    if data.get('timeline'):
        try:
            dates = [event.get('date') for event in data['timeline'] if event.get('date')]
            if dates:
                order_date = max(dates)
        except Exception as e:
            logger.warning(f"Error extracting order date: {e}")

    # Prepare order entry
    order_entry = {
        'thread_id': data.get('thread_id'),
        'date': order_date,
        'total_amount': total_amount,
        'purchase_order': order_numbers.get('purchase_order'),
        'sales_invoice': order_numbers.get('sales_invoice'),
        'reference_numbers': order_numbers.get('reference_numbers', []),
        'order_numbers': order_numbers.get('order_numbers', []),
        'is_reseller_order': relationship.get('is_reseller_order', False)
    }

    # Add reseller/end customer info if applicable
    if relationship.get('is_reseller_order'):
        order_entry['reseller'] = relationship.get('reseller')
        order_entry['end_customer'] = relationship.get('end_customer')

    return order_entry


def safe_get_total_amount(data):
    """
    Safely extract total amount from quote details.
    Returns 0 if no valid total amount can be found.
    """
    # If data is None, return 0
    if data is None:
        return 0

    # If quote_details is None, use the data directly if it looks like a number
    quote_details = data.get('quote_details') or {}
    total_amount = quote_details.get('total_amount', 0)

    # If total_amount is None, try to find a number in the dict
    if total_amount is None:
        # Look for keys that might contain amount information
        amount_keys = ['total', 'amount', 'value', 'quote_value']
        for key in amount_keys:
            potential_amount = quote_details.get(key)
            if potential_amount is not None:
                try:
                    return float(potential_amount)
                except (ValueError, TypeError):
                    continue

        # If no amount found, return 0
        return 0

    # Try to convert to float, default to 0 if not possible
    try:
        return float(total_amount)
    except (ValueError, TypeError):
        return 0


def match_company_names(name1, name2, threshold=80):
    """Match company names using fuzzy string matching."""
    if not name1 or not name2:
        return False

    # Normalize both names
    norm1 = normalize_company_name(name1)
    norm2 = normalize_company_name(name2)

    if not norm1 or not norm2:
        return False

    # Use fuzzywuzzy for string matching
    similarity = fuzz.token_sort_ratio(norm1, norm2)
    return similarity >= threshold


def normalize_product_name(name):
    """Normalize product name for better matching."""
    if not name:
        return None

    # Convert to lowercase and remove common words
    normalized = name.lower()

    # Remove model numbers and specific identifiers that might vary
    normalized = re.sub(r'\b(model|v|ver|version|rev)\s*[\d\.]+\b', '', normalized)

    # Remove common product suffixes and prefixes
    prefixes = ['strike ', 'new ', 'premium ']
    suffixes = [' series', ' model', ' edition', ' collection']

    for prefix in prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]

    # Remove punctuation and extra whitespace
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = ' '.join(normalized.split())

    return normalized


def is_payout_confirmation(data):
    """
    Check if the conversation is a payout confirmation.
    Returns True if it is a payout confirmation, False otherwise.
    """
    # Check for payout keywords in timeline events
    if 'timeline' in data:
        for event in data.get('timeline', []):
            if event is None:
                continue
            # If the event is a dict, get the 'event' key; otherwise, treat the event itself as text.
            if isinstance(event, dict):
                event_text = event.get('event', '')
            else:
                event_text = str(event)
            if not event_text:
                continue
            event_text_lower = event_text.lower()
            if any(keyword in event_text_lower for keyword in
                   ['payout confirmation', 'payment confirmed', 'funds transferred',
                    'payment processed', 'remittance', 'bank transfer']):
                return True

    # Check conversation subject if available
    if 'subject' in data:
        subject = data.get('subject', '')
        if subject is not None:  # Additional check to ensure subject is not None
            subject_lower = subject.lower()
            if any(keyword in subject_lower for keyword in ['payout', 'payment', 'transfer', 'remittance']):
                return True

    # Check for payout keywords in entities
    if 'entities' in data:
        for entity in data.get('entities', []):
            # Skip None entities
            if entity is None:
                continue

            # Skip if entity is not a dictionary
            if not isinstance(entity, dict):
                continue

            # Check entity type
            entity_type = entity.get('type', '')
            if entity_type is not None:  # Check if entity_type is not None
                entity_type_lower = entity_type.lower()
                if 'payment' in entity_type_lower or 'transfer' in entity_type_lower:
                    return True

            # Check description
            description = entity.get('description', '')
            if description is not None:  # Check if description is not None
                description_lower = description.lower()
                if any(keyword in description_lower for keyword in
                       ['payout', 'payment confirmation', 'transfer confirmation']):
                    return True

    return False


# To use this in the main processing loop:
# for thread_id, data in tqdm(detailed_results.items(), desc="Processing clients"):
#     # Skip payout confirmations
#     if is_payout_confirmation(data):
#         logger.info(f"Skipping payout confirmation thread {thread_id}")
#         continue

def extract_order_numbers(data):
    """
    Extract all possible order and reference numbers from conversation data.
    Returns a dictionary of different order number types found.
    """
    order_numbers = {
        'purchase_order': None,
        'sales_invoice': None,
        'reference_numbers': [],
        'order_numbers': []
    }

    # Check quote details for standard PO and invoice
    if 'quote_details' in data:
        quote_details = data.get('quote_details')

        # Make sure quote_details is a dictionary
        if quote_details is not None and isinstance(quote_details, dict):
            if quote_details.get('purchase_order'):
                order_numbers['purchase_order'] = quote_details.get('purchase_order')
            if quote_details.get('sales_invoice'):
                order_numbers['sales_invoice'] = quote_details.get('sales_invoice')

    # Rest of the function remains the same...
    # Look for additional order references in the timeline
    if 'timeline' in data:
        for event in data['timeline']:
            if not isinstance(event, dict):
                continue

            event_text = event.get('event', '')
            if event_text is None:
                continue

            event_text = str(event_text).lower()

            # Look for order number patterns
            order_patterns = [
                r'order\s+(?:number|#|no\.?|id)?\s*[:#]?\s*(\w+[-\d]+\w*)',
                r'order\s+(?:placed|confirmed|received).*?(\w+[-\d]+\w*)',
                r'(?:po|purchase order)\s+(?:number|#|no\.?|id)?\s*[:#]?\s*(\w+[-\d]+\w*)',
                r'reference\s+(?:number|#|no\.?|id)?\s*[:#]?\s*(\w+[-\d]+\w*)',
                r'ref\s+(?:number|#|no\.?|id)?\s*[:#]?\s*(\w+[-\d]+\w*)'
            ]

            for pattern in order_patterns:
                matches = re.findall(pattern, event_text)
                for match in matches:
                    if re.search(r'\d', match):  # Ensure it contains at least one digit
                        if 'reference' in event_text or 'ref' in event_text:
                            if match not in order_numbers['reference_numbers']:
                                order_numbers['reference_numbers'].append(match)
                        else:
                            if match not in order_numbers['order_numbers']:
                                order_numbers['order_numbers'].append(match)

    # Look in entity descriptions for more order references
    if 'entities' in data:
        for entity in data['entities']:
            if not isinstance(entity, dict):
                continue

            if entity.get('type') == 'Product' and entity.get('description'):
                desc = entity.get('description', '')
                if desc is None:
                    continue

                desc = str(desc).lower()

                # Look for embedded reference numbers (often in *** brackets)
                ref_patterns = [
                    r'\*\*\*(\w+[-\d]+\w*)\*\*\*',
                    r'ref(?:erence)?[:#]?\s*(\w+[-\d]+\w*)',
                    r'order[:#]?\s*(\w+[-\d]+\w*)'
                ]

                for pattern in ref_patterns:
                    matches = re.findall(pattern, desc)
                    for match in matches:
                        if re.search(r'\d', match) and match not in order_numbers['reference_numbers']:
                            order_numbers['reference_numbers'].append(match)

    return order_numbers


def detect_reseller_relationship(data):
    """
    Detect if a conversation involves a reseller relationship.
    Returns a dict with reseller and end customer information.
    """
    relationship = {
        'is_reseller_order': False,
        'reseller': None,
        'end_customer': None,
        'confidence': 0
    }

    # Not enough data to determine
    if 'entities' not in data or len(data['entities']) < 2:
        return relationship

    # Look for multiple companies in the entities
    companies = []

    for entity in data['entities']:
        # Skip None entities
        if entity is None:
            continue

        if entity.get('company') and entity.get('company') != 'Unknown' and entity.get('company') != 'Strike':
            if entity.get('entity') != entity.get('company'):  # Avoid duplicating when entity name is same as company
                companies.append({
                    'name': entity.get('company'),
                    'role': entity.get('role', 'Unknown'),
                    'entity_type': 'company' if 'contact_info' not in entity else 'person'
                })

    # Rest of the function remains the same...
    # (including the existing code for checking multiple companies, etc.)

    # If we have multiple companies, likely a reseller relationship
    if len(companies) >= 2:
        relationship['is_reseller_order'] = True
        relationship['confidence'] = 0.8

        # Try to determine which is reseller and which is end customer
        deliver_to_keywords = ['deliver to', 'ship to', 'destination', 'end user']
        ordered_by_keywords = ['ordered by', 'bill to', 'sold to', 'supplier']

        # Check timeline for clues
        if 'timeline' in data:
            for event in data['timeline']:
                event_text = event.get('event', '').lower()

                for company in companies:
                    company_name = company['name'].lower()

                    # Check if this company is mentioned as the delivery destination
                    if any(keyword in event_text and company_name in event_text for keyword in deliver_to_keywords):
                        relationship['end_customer'] = company['name']
                        relationship['confidence'] = 0.9

                    # Check if this company is mentioned as the orderer
                    if any(keyword in event_text and company_name in event_text for keyword in ordered_by_keywords):
                        relationship['reseller'] = company['name']
                        relationship['confidence'] = 0.9

        # If we still don't know, make educated guesses based on known resellers
        known_resellers = ['jb hi-fi', 'ingram micro', 'synnex', 'dicker data', 'leader computers']

        if not relationship['reseller'] or not relationship['end_customer']:
            for company in companies:
                company_norm = company['name'].lower()

                # Check if it's a known reseller
                if any(reseller in company_norm for reseller in known_resellers):
                    relationship['reseller'] = company['name']
                    # Find another company to be the end customer
                    for other_company in companies:
                        if other_company['name'] != company['name']:
                            relationship['end_customer'] = other_company['name']
                            relationship['confidence'] = 0.85
                            break
                    break

        # If we still don't have both, just pick the first two
        if not relationship['reseller'] and len(companies) > 0:
            relationship['reseller'] = companies[0]['name']

        if not relationship['end_customer'] and len(companies) > 1:
            # Make sure we don't duplicate
            for company in companies:
                if company['name'] != relationship['reseller']:
                    relationship['end_customer'] = company['name']
                    break

    return relationship


def extract_company_from_entity(entities_list):
    """Extract the most likely client company from entities list."""
    if not entities_list:
        return None

    # First pass: look for entities with both company and contact info
    for entity in entities_list:
        # Skip None entities
        if entity is None:
            continue

        # Skip if entity is not a dictionary
        if not isinstance(entity, dict):
            continue

        if entity.get('entity') and entity.get('company') and entity.get('company') != 'Unknown':
            # Check if contact_info exists, is a dictionary, and has an email
            contact_info = entity.get('contact_info')
            if contact_info is not None and isinstance(contact_info, dict) and contact_info.get('email'):
                return entity.get('company')

    # Second pass: any entity with a company
    for entity in entities_list:
        # Skip None entities
        if entity is None:
            continue

        # Skip if entity is not a dictionary
        if not isinstance(entity, dict):
            continue

        if entity.get('company') and entity.get('company') != 'Unknown':
            return entity.get('company')

    return None


def extract_products_from_entity(entities_list):
    """Extract products from entities list."""
    products = []
    if not entities_list:
        return products

    for entity in entities_list:
        # Skip None entities
        if entity is None:
            continue

        # Skip if entity is not a dictionary
        if not isinstance(entity, dict):
            continue

        if entity.get('type') == 'Product':
            product = {
                'name': entity.get('entity', 'Unknown Product'),
                'description': entity.get('description', ''),
                'sku': entity.get('sku'),
                'price': parse_price(entity.get('price')),
                'quantity': entity.get('quantity')
            }

            # Ensure product description is not None
            if product['description'] is None:
                product['description'] = ''

            products.append(product)

    return products


def parse_price(price_str):
    """Extract numeric price from string."""
    if not price_str:
        return None

    # Remove currency symbols and commas
    cleaned = re.sub(r'[^\d.]', '', str(price_str))

    try:
        return float(cleaned)
    except:
        return None


def infer_client_industry(client_data):
    """Infer client industry based on patterns in company name and product purchases."""
    if not client_data:
        return 'Unknown'

    company_name = (client_data.get('company_name') or '').lower()
    products = client_data.get('products', [])
    product_names = [p.get('name', '').lower() for p in products]
    product_descs = [p.get('description', '').lower() for p in products]

    # Industry keywords
    industries = {
        'Healthcare': ['hospital', 'medical', 'health', 'clinic', 'doctor', 'pharma', 'healthcare'],
        'Construction': ['construction', 'builder', 'building', 'contractor', 'project', 'site'],
        'Logistics': ['logistics', 'transport', 'shipping', 'delivery', 'freight', 'cargo'],
        'Mining': ['mining', 'resource', 'mineral', 'excavation', 'drill'],
        'Manufacturing': ['manufacturing', 'factory', 'industrial', 'production', 'assembly'],
        'Retail': ['retail', 'store', 'shop', 'mall', 'outlet', 'market'],
        'IT & Technology': ['technology', 'computer', 'software', 'hardware', 'it ', 'tech'],
        'Government': ['government', 'council', 'department', 'agency', 'authority', 'public'],
        'Education': ['school', 'university', 'college', 'education', 'academy', 'institute'],
        'Telecommunications': ['telecom', 'communication', 'network', 'mobile', 'phone'],
        'Reseller/Distributor': ['retail group', 'distributor', 'reseller', 'dealer', 'var ', 'channel partner',
                                 'wholesale']
    }

    # Check if it's a known reseller
    known_resellers = ['jb hi-fi', 'ingram micro', 'synnex', 'dicker data', 'leader computers']
    if any(reseller in company_name for reseller in known_resellers):
        return 'Reseller/Distributor'

    # Check company name for industry keywords
    for industry, keywords in industries.items():
        if any(keyword in company_name for keyword in keywords):
            return industry

    # Check product descriptions for industry clues
    for industry, keywords in industries.items():
        if any(any(keyword in desc for keyword in keywords) for desc in product_descs):
            return industry

    # Check for patterns in purchased products
    rugged_products = sum(1 for p in product_names if 'rugged' in p or 'tough' in p)
    vehicle_products = sum(1 for p in product_names if 'car' in p or 'vehicle' in p or 'truck' in p)

    if rugged_products > 2 and vehicle_products > 1:
        return 'Field Services'

    return 'Other'


# === CLIENT CATEGORIZATION FUNCTIONS ===
def categorize_by_client(detailed_results, comprehensive_results):
    """
    Categorize and combine conversations by client.
    Returns a structured client database with all related conversations and orders.
    """
    print("Starting client categorization...")
    logger.info("Starting client categorization process.")

    # Initialize client database
    client_db = {}

    # Initialize reseller relationships database
    reseller_relationships = {}

    # Process each conversation and organize by client
    for thread_id, data in tqdm(detailed_results.items(), desc="Processing clients"):
        # Skip if data is None or not a dictionary
        if not data or not isinstance(data, dict):
            logger.warning(f"Skipping thread {thread_id} because data is invalid: {data}")
            continue

        # Skip payout confirmations
        if is_payout_confirmation(data):
            logger.info(f"Skipping payout confirmation thread {thread_id}")
            continue

        # Extract client information
        client_company = extract_company_from_entity(data.get('entities', []))
        client_email = None

        for entity in data.get('entities', []):
            # Skip None entities
            if entity is None:
                continue

            # Skip if entity is not a dictionary
            if not isinstance(entity, dict):
                continue

            contact_info = entity.get('contact_info')

            # Skip if contact_info is None
            if contact_info is None:
                continue

            # Handle string contact_info
            if isinstance(contact_info, str):
                # If it looks like an email address
                if '@' in contact_info and 'strike.com' not in contact_info:
                    client_email = contact_info
                    break
                continue

            # Handle dict contact_info (expected case)
            if isinstance(contact_info, dict):
                email = contact_info.get('email')
                if email and 'strike.com' not in email:
                    client_email = email
                    break

        # If no email found in entities, try to get from comprehensive analysis
        if not client_email and thread_id in comprehensive_results:
            initiator_email = comprehensive_results[thread_id].get('initiator_email')
            if initiator_email and 'strike.com' not in initiator_email:
                client_email = initiator_email

        # Skip if we can't identify the client
        if not client_company and not client_email:
            logger.warning(f"Could not identify client for thread {thread_id}")
            continue

        # Extract domain from email for grouping
        email_domain = extract_domain(client_email) if client_email else None

        # Check for reseller relationship
        relationship = detect_reseller_relationship(data)

        # If this is a reseller order, we need to track both reseller and end customer
        if relationship['is_reseller_order'] and relationship['reseller'] and relationship['end_customer']:
            # Store the relationship
            relationship_key = f"{relationship['reseller']}_{relationship['end_customer']}"
            if relationship_key not in reseller_relationships:
                reseller_relationships[relationship_key] = {
                    'reseller': relationship['reseller'],
                    'end_customer': relationship['end_customer'],
                    'orders': [],
                    'first_order_date': None,
                    'last_order_date': None,
                    'total_order_value': 0
                }

            # Extract order details and numbers
            order_numbers = extract_order_numbers(data)
            order_date = None

            # Try to find order date from timeline
            if 'timeline' in data:
                # Look for the most recent date in the timeline
                dates = [event.get('date') for event in data.get('timeline', []) if event.get('date')]
                if dates:
                    order_date = max(dates)

            # Add order to relationship
            if order_date or order_numbers['purchase_order'] or order_numbers['sales_invoice']:
                quote_details = data.get('quote_details') or {}
                total_amount = quote_details.get('total_amount', 0)
                if total_amount is None:
                    order_value = 0
                elif isinstance(total_amount, str):
                    # Try to convert string to float
                    try:
                        order_value = float(total_amount)
                    except (ValueError, TypeError):
                        order_value = 0
                else:
                    order_value = total_amount or 0

                reseller_relationships[relationship_key]['orders'].append({
                    'thread_id': thread_id,
                    'date': order_date,
                    'purchase_order': order_numbers['purchase_order'],
                    'sales_invoice': order_numbers['sales_invoice'],
                    'reference_numbers': order_numbers['reference_numbers'],
                    'order_numbers': order_numbers['order_numbers'],
                    'order_value': order_value
                })

                # Update relationship stats
                if order_value:
                    reseller_relationships[relationship_key]['total_order_value'] += order_value

                if order_date:
                    if not reseller_relationships[relationship_key]['first_order_date'] or order_date < \
                            reseller_relationships[relationship_key]['first_order_date']:
                        reseller_relationships[relationship_key]['first_order_date'] = order_date

                    if not reseller_relationships[relationship_key]['last_order_date'] or order_date > \
                            reseller_relationships[relationship_key]['last_order_date']:
                        reseller_relationships[relationship_key]['last_order_date'] = order_date

            # For reseller orders, we want to track both entities
            # First, let's handle the reseller
            primary_client = relationship['reseller']
        else:
            # For regular orders, use the extracted client company
            primary_client = client_company

        # Create a client identifier
        client_key = None

        # First try to find existing client by exact company name
        if primary_client:
            for key in client_db:
                existing_company = client_db[key].get('company_name')
                if existing_company and primary_client.lower() == existing_company.lower():
                    client_key = key
                    break

        # Then try fuzzy matching on company names
        if not client_key and primary_client:
            for key in client_db:
                existing_company = client_db[key].get('company_name')
                if match_company_names(primary_client, existing_company):
                    client_key = key
                    # Update with the more specific company name if available
                    if primary_client and len(primary_client) > len(existing_company):
                        client_db[key]['company_name'] = primary_client
                    break

        # Then try matching by email domain
        if not client_key and email_domain:
            for key in client_db:
                if email_domain == client_db[key].get('email_domain'):
                    client_key = key
                    # Update company name if we now have one
                    if primary_client and not client_db[key].get('company_name'):
                        client_db[key]['company_name'] = primary_client
                    break

        # If still no match, create a new client entry
        if not client_key:
            client_key = f"client_{len(client_db) + 1}"
            client_db[client_key] = {
                'company_name': primary_client,
                'email_domain': email_domain,
                'contacts': [],
                'conversations': [],
                'orders': [],
                'products': [],
                'first_contact': None,
                'last_contact': None,
                'is_reseller': relationship['is_reseller_order'] and primary_client == relationship['reseller'],
                'is_end_customer': relationship['is_reseller_order'] and primary_client == relationship['end_customer'],
                'reseller_relationships': [],
                'end_customer_relationships': []
            }

        # Update client record
        client_record = client_db[client_key]

        # If this is a reseller, link to the end customer
        if relationship['is_reseller_order'] and relationship['reseller'] and relationship['end_customer']:
            # For reseller, track their end customers
            if primary_client == relationship['reseller'] and relationship['end_customer'] not in client_record[
                'end_customer_relationships']:
                client_record['end_customer_relationships'].append(relationship['end_customer'])

            # If we're also adding the end customer separately, track their reseller
            if primary_client == relationship['end_customer'] and relationship['reseller'] not in client_record[
                'reseller_relationships']:
                client_record['reseller_relationships'].append(relationship['reseller'])


        # Add conversation reference
        conversation_entry = {
            'thread_id': thread_id,
            'quote_details': data.get('quote_details', {}),
            'timeline': data.get('timeline', []),
            'is_reseller_order': relationship['is_reseller_order']
        }

        # Add detailed order numbers
        order_numbers = extract_order_numbers(data)
        conversation_entry['order_numbers'] = order_numbers

        # Add reseller relationship if applicable
        if relationship['is_reseller_order']:
            conversation_entry['reseller_relationship'] = {
                'reseller': relationship['reseller'],
                'end_customer': relationship['end_customer']
            }

        # Add communication statistics if available
        if thread_id in comprehensive_results:
            conv_analysis = comprehensive_results[thread_id].get('conversation_analysis', {})
            conversation_entry.update({
                'sentiment': conv_analysis.get('overall_sentiment', 'unknown'),
                'outcome': conv_analysis.get('outcome', 'unknown'),
                'sales_cycle_stage': conv_analysis.get('sales_cycle_stage', 'unknown')
            })

        client_record['conversations'].append(conversation_entry)

        # Extract products from this conversation
        products = extract_products_from_entity(data.get('entities', []))
        for product in products:
            # Check if product already exists for this client
            exists = False
            for existing_product in client_record['products']:
                if normalize_product_name(product.get('name')) == normalize_product_name(existing_product.get('name')):
                    exists = True
                    # Update with more details if available
                    if product.get('sku') and not existing_product.get('sku'):
                        existing_product['sku'] = product.get('sku')

                    # Safely compare description lengths, handling None values
                    product_desc = product.get('description', '')
                    existing_desc = existing_product.get('description', '')

                    # Convert None to empty string for safe comparison
                    product_desc = '' if product_desc is None else product_desc
                    existing_desc = '' if existing_desc is None else existing_desc

                    if len(product_desc) > len(existing_desc):
                        existing_product['description'] = product_desc
                    break

            if not exists:
                # Ensure product description is not None
                if product.get('description') is None:
                    product['description'] = ''
                client_record['products'].append(product)

        # Get quote status safely, handling None values
        quote_details = data.get('quote_details') or {}
        quote_status_value = quote_details.get('quote_status', '')
        # Convert None to empty string if necessary
        order_status = '' if quote_status_value is None else quote_status_value.lower()

        if order_status in ['accepted', 'completed'] or (data.get('quote_details') or {}).get('total_amount'):
            # Order processing code...
            order_entry = {
                'thread_id': thread_id,
                'date': next((event.get('date') for event in data.get('timeline', []) if
                      'purchase' in event.get('event', '').lower() or 'accept' in event.get('event', '').lower()), None),
                'total_amount': (data.get('quote_details') or {}).get('total_amount'),
                'purchase_order': (data.get('quote_details') or {}).get('purchase_order'),
                'sales_invoice': (data.get('quote_details') or {}).get('sales_invoice'),
                'reference_numbers': order_numbers['reference_numbers'],
                'order_numbers': order_numbers['order_numbers'],
                'products': products,
                'is_reseller_order': relationship['is_reseller_order']
            }

            # Add reseller/end customer info if applicable
            if relationship['is_reseller_order']:
                order_entry['reseller'] = relationship['reseller']
                order_entry['end_customer'] = relationship['end_customer']

            client_record['orders'].append(order_entry)

        # Update first/last contact dates
        dates = [event.get('date') for event in data.get('timeline', []) if event.get('date')]
        if dates:
            dates.sort()
            if not client_record['first_contact'] or dates[0] < client_record['first_contact']:
                client_record['first_contact'] = dates[0]
            if not client_record['last_contact'] or dates[-1] > client_record['last_contact']:
                client_record['last_contact'] = dates[-1]

        # If this was a reseller order, also process the end customer if different
        if relationship['is_reseller_order'] and relationship['reseller'] and relationship['end_customer']:
            if relationship['reseller'] != relationship['end_customer'] and primary_client == relationship['reseller']:
                # We need to add/update the end customer record too
                end_customer_name = relationship['end_customer']

                # Look for existing end customer record
                end_customer_key = None
                for key in client_db:
                    existing_company = client_db[key].get('company_name')
                    if existing_company and (
                            end_customer_name.lower() == existing_company.lower() or match_company_names(
                        end_customer_name, existing_company)):
                        end_customer_key = key
                        break

                # Create new end customer record if needed
                if not end_customer_key:
                    end_customer_key = f"client_{len(client_db) + 1}"
                    client_db[end_customer_key] = {
                        'company_name': end_customer_name,
                        'email_domain': None,
                        'contacts': [],
                        'conversations': [],
                        'orders': [],
                        'products': [],
                        'first_contact': None,
                        'last_contact': None,
                        'is_reseller': False,
                        'is_end_customer': True,
                        'reseller_relationships': [relationship['reseller']],
                        'end_customer_relationships': []
                    }

                # Add this conversation to the end customer
                client_db[end_customer_key]['conversations'].append(conversation_entry)

                # Add products
                for product in products:
                    # Check if product already exists for this client
                    exists = False
                    for existing_product in client_db[end_customer_key]['products']:
                        if normalize_product_name(product.get('name')) == normalize_product_name(
                                existing_product.get('name')):
                            exists = True
                            break

                    if not exists:
                        client_db[end_customer_key]['products'].append(product)

                # Add order if applicable
                if order_status in ['accepted', 'completed'] or (data.get('quote_details') or {}).get('total_amount'):
                    client_db[end_customer_key]['orders'].append(order_entry)

                # Update first/last contact dates
                if dates:
                    if not client_db[end_customer_key]['first_contact'] or dates[0] < client_db[end_customer_key][
                        'first_contact']:
                        client_db[end_customer_key]['first_contact'] = dates[0]
                    if not client_db[end_customer_key]['last_contact'] or dates[-1] > client_db[end_customer_key][
                        'last_contact']:
                        client_db[end_customer_key]['last_contact'] = dates[-1]

    # Calculate derived metrics for each client
    for client_key, client_data in client_db.items():
        # Calculate total spend
        total_spend = sum(safe_to_float(order.get('total_amount', 0)) for order in client_data.get('orders', []))
        client_data['total_spend'] = total_spend

        # Calculate average order value
        if client_data.get('orders'):
            client_data['average_order_value'] = total_spend / len(client_data.get('orders')) if total_spend else 0
        else:
            client_data['average_order_value'] = 0

        # Success rate
        quote_count = len([c for c in client_data.get('conversations', []) if c.get('quote_details')])
        order_count = len(client_data.get('orders', []))
        client_data['quote_to_order_rate'] = round((order_count / quote_count * 100), 2) if quote_count > 0 else 0

        # Infer industry and size
        client_data['industry'] = infer_client_industry(client_data)

        # Flag known resellers by industry
        if client_data['industry'] == 'Reseller/Distributor':
            client_data['is_reseller'] = True

        # Determine business size
        client_data['size'] = determine_business_size(client_data)

    # Save reseller relationships
    save_results(reseller_relationships, RESELLER_RELATIONSHIPS_FILE)

    print(f"Client categorization complete. Identified {len(client_db)} unique clients.")
    logger.info(f"Client categorization complete. Identified {len(client_db)} unique clients.")

    return client_db


def determine_business_size(client_data):
    """Determine the business size based on order history and other factors."""
    # For resellers, look at their volume
    if client_data.get('is_reseller', False):
        total_spent = client_data.get('total_spend', 0)
        num_end_customers = len(client_data.get('end_customer_relationships', []))

        if total_spent > 50000 or num_end_customers > 10:
            return 'Large'
        elif total_spent > 10000 or num_end_customers > 5:
            return 'Medium'
        else:
            return 'Small'

    # For regular customers
    total_spent = client_data.get('total_spend', 0)
    avg_order_value = client_data.get('average_order_value', 0)
    order_frequency = len(client_data.get('orders', [])) / 12  # Orders per month

    # Size classification based on spending patterns
    if total_spent > 50000 or avg_order_value > 10000:
        return 'Enterprise'
    elif total_spent > 10000 or (avg_order_value > 2000 and order_frequency > 0.5):
        return 'Large'
    elif total_spent > 2000 or avg_order_value > 500:
        return 'Medium'
    else:
        return 'Small'


# === PRODUCT CATEGORIZATION FUNCTIONS ===
def categorize_by_product(detailed_results, comprehensive_results, client_db):
    """
    Categorize and combine conversations by product.
    Returns a structured product database with all related quotes and orders.
    """
    print("Starting product categorization...")
    logger.info("Starting product categorization process.")

    # Initialize product database
    product_db = {}

    # Process each conversation and organize by product
    for thread_id, data in tqdm(detailed_results.items(), desc="Processing products"):
        # Extract products from this conversation
        raw_products = extract_products_from_entity(data.get('entities', []))

        # Skip if no products found
        if not raw_products:
            continue

        # Extract quote details
        quote_status = (data.get('quote_details') or {}).get('quote_status', 'unknown')
        quote_amount = (data.get('quote_details') or {}).get('total_amount')

        # Extract client info
        client_company = extract_company_from_entity(data.get('entities', []))

        # Check for reseller relationship
        relationship = detect_reseller_relationship(data)

        # Process each product
        for product in raw_products:
            product_name = product.get('name')
            if not product_name:
                continue

            # Try to find matching product in database
            product_key = None
            for key in product_db:
                if normalize_product_name(product_name) == normalize_product_name(product_db[key].get('name')):
                    product_key = key
                    # Update with better name if available
                    if len(product_name) > len(product_db[key].get('name', '')):
                        product_db[key]['name'] = product_name
                    break

            # If no match found, create new product entry
            if not product_key:
                product_key = f"product_{len(product_db) + 1}"
                category = categorize_product(product)

                product_db[product_key] = {
                    'name': product_name,
                    'sku': product.get('sku'),
                    'description': product.get('description', ''),
                    'category': category,
                    'quotes': [],
                    'orders': [],
                    'price_points': [],
                    'clients': set(),
                    'reseller_orders': [],
                    'direct_orders': []
                }

            # Add SKU if not already present
            if not product_db[product_key]['sku'] and product.get('sku'):
                product_db[product_key]['sku'] = product.get('sku')

            # Add quote/order information
            quote_entry = {
                'thread_id': thread_id,
                'client': client_company,
                'price': product.get('price'),
                'quantity': product.get('quantity'),
                'status': quote_status,
                'is_reseller_order': relationship['is_reseller_order']
            }

            # Add reseller relationship if applicable
            if relationship['is_reseller_order']:
                quote_entry['reseller'] = relationship['reseller']
                quote_entry['end_customer'] = relationship['end_customer']

            # Add to quotes or orders based on status
            if quote_status in ['accepted', 'completed']:
                product_db[product_key]['orders'].append(quote_entry)

                # Also track by order type for analysis
                if relationship['is_reseller_order']:
                    product_db[product_key]['reseller_orders'].append(quote_entry)
                else:
                    product_db[product_key]['direct_orders'].append(quote_entry)
            else:
                product_db[product_key]['quotes'].append(quote_entry)

            # Add price point if available
            if product.get('price'):
                price_entry = {
                    'price': product.get('price'),
                    'quantity': product.get('quantity'),
                    'date': next((event.get('date') for event in data.get('timeline', []) if event.get('date')), None),
                    'client': client_company,
                    'is_reseller': relationship['is_reseller_order'],
                    'order_status': quote_status
                }

                # Add end customer if reseller
                if relationship['is_reseller_order']:
                    price_entry['reseller'] = relationship['reseller']
                    price_entry['end_customer'] = relationship['end_customer']

                product_db[product_key]['price_points'].append(price_entry)

            # Add client to set of clients for this product
            if client_company:
                product_db[product_key]['clients'].add(client_company)

                # Also add end customer if reseller order
                if relationship['is_reseller_order'] and relationship['end_customer']:
                    product_db[product_key]['clients'].add(relationship['end_customer'])

    # Calculate derived metrics for each product
    for product_key, product_data in product_db.items():
        # Convert clients set to list for JSON serialization
        product_data['clients'] = list(product_data['clients'])

        # Calculate quote to order conversion rate
        quote_count = len(product_data.get('quotes', []))
        order_count = len(product_data.get('orders', []))
        total_inquiries = quote_count + order_count

        product_data['conversion_rate'] = round((order_count / total_inquiries * 100), 2) if total_inquiries > 0 else 0
        product_data['total_inquiries'] = total_inquiries
        product_data['total_orders'] = order_count

        # Calculate separate metrics for reseller vs direct
        reseller_count = len(product_data.get('reseller_orders', []))
        direct_count = len(product_data.get('direct_orders', []))

        if reseller_count > 0:
            product_data['reseller_order_percentage'] = round((reseller_count / order_count * 100),
                                                              2) if order_count > 0 else 0

        # Calculate average price, if price points exist
        prices = [p.get('price') for p in product_data.get('price_points', []) if p.get('price')]
        if prices:
            product_data['average_price'] = sum(prices) / len(prices)
            product_data['price_range'] = {
                'min': min(prices),
                'max': max(prices)
            }

            # Calculate separate price metrics for reseller vs direct orders
            reseller_prices = [p.get('price') for p in product_data.get('price_points', []) if
                               p.get('price') and p.get('is_reseller', False)]
            direct_prices = [p.get('price') for p in product_data.get('price_points', []) if
                             p.get('price') and not p.get('is_reseller', False)]

            if reseller_prices:
                product_data['reseller_average_price'] = sum(reseller_prices) / len(reseller_prices)

            if direct_prices:
                product_data['direct_average_price'] = sum(direct_prices) / len(direct_prices)

        # Calculate total units sold
        total_units = sum(order.get('quantity', 0) or 0 for order in product_data.get('orders', []))
        product_data['total_units_sold'] = total_units

        # Calculate total revenue
        total_revenue = sum(
            (order.get('price', 0) or 0) * (order.get('quantity', 0) or 0) for order in product_data.get('orders', []))
        product_data['total_revenue'] = total_revenue

        # Calculate client industry distribution
        industry_distribution = defaultdict(int)
        for client_name in product_data['clients']:
            # Find client in client_db
            client_match = None
            for client_key, client_info in client_db.items():
                if match_company_names(client_name, client_info.get('company_name', '')):
                    client_match = client_info
                    break

            if client_match:
                industry = client_match.get('industry', 'Unknown')
                industry_distribution[industry] += 1

        product_data['industry_distribution'] = dict(industry_distribution)

    print(f"Product categorization complete. Identified {len(product_db)} unique products.")
    logger.info(f"Product categorization complete. Identified {len(product_db)} unique products.")

    return product_db


def categorize_product(product):
    """Categorize product based on name and description."""
    name = product.get('name', '').lower()
    description = product.get('description', '').lower()

    # Category keywords
    categories = {
        'Phone Case': ['case', 'cover', 'protection', 'screen protector'],
        'Vehicle Mount': ['mount', 'cradle', 'holder', 'dock', 'car', 'vehicle', 'dashboard'],
        'Charging Solution': ['charger', 'charging', 'power', 'cable', 'adapter', 'battery'],
        'Signal Booster': ['signal', 'antenna', 'repeater', 'booster', 'amplifier', 'reception'],
        'Tablet Accessory': ['tablet', 'ipad', 'galaxy tab', 'surface']
    }

    # Check product name and description for category keywords
    for category, keywords in categories.items():
        if any(keyword in name for keyword in keywords) or any(keyword in description for keyword in keywords):
            return category

    return 'Other'


# === COMBINED ANALYSIS FUNCTIONS ===
def perform_combined_analysis(client_db, product_db, reseller_relationships):
    """
    Perform combined analysis of client and product data.
    Identifies patterns, correlations, and generates insights.
    """
    print("Starting combined analysis...")
    logger.info("Starting combined analysis process.")

    # Initialize results structure
    analysis_results = {
        'client_segments': [],
        'product_insights': [],
        'cross_sell_opportunities': [],
        'pricing_insights': [],
        'industry_product_matrix': {},
        'reseller_insights': [],
        'general_insights': []
    }

    # === CLIENT SEGMENT ANALYSIS ===
    # Group clients by industry and size
    industry_size_segments = defaultdict(list)
    for client_id, client_data in client_db.items():
        segment_key = f"{client_data.get('industry', 'Unknown')}-{client_data.get('size', 'Unknown')}"
        industry_size_segments[segment_key].append(client_id)

    # Analyze each segment
    for segment_key, client_ids in industry_size_segments.items():
        if len(client_ids) < 2:  # Skip segments with only one client
            continue

        industry, size = segment_key.split('-')

        # Collect segment metrics
        total_spend = sum(client_db[client_id].get('total_spend', 0) for client_id in client_ids)
        avg_order_value = sum(client_db[client_id].get('average_order_value', 0) for client_id in client_ids) / len(
            client_ids)
        conversion_rate = sum(client_db[client_id].get('quote_to_order_rate', 0) for client_id in client_ids) / len(
            client_ids)

        # Collect common products
        all_products = []
        for client_id in client_ids:
            all_products.extend(client_db[client_id].get('products', []))

        product_counts = defaultdict(int)
        for product in all_products:
            product_counts[product.get('name')] += 1

        top_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        # Generate segment insight
        segment_insight = {
            'segment': segment_key,
            'industry': industry,
            'size': size,
            'client_count': len(client_ids),
            'total_spend': total_spend,
            'average_order_value': avg_order_value,
            'average_conversion_rate': conversion_rate,
            'top_products': [{'name': p[0], 'count': p[1]} for p in top_products],
            'client_examples': client_ids[:3]  # Sample clients
        }

        analysis_results['client_segments'].append(segment_insight)

    # Sort client segments by total spend
    analysis_results['client_segments'].sort(key=lambda x: x.get('total_spend', 0), reverse=True)

    # === RESELLER INSIGHTS ===
    # Identify and analyze resellers
    resellers = [client_id for client_id, data in client_db.items() if data.get('is_reseller', False)]

    if resellers:
        # Analyze each reseller
        for reseller_id in resellers:
            reseller_data = client_db[reseller_id]
            reseller_name = reseller_data.get('company_name')

            # Find all orders where this company is the reseller
            reseller_orders = []
            for rel_key, rel_data in reseller_relationships.items():
                if match_company_names(reseller_name, rel_data.get('reseller', '')):
                    reseller_orders.extend(rel_data.get('orders', []))

            # Calculate reseller metrics
            total_reseller_revenue = sum(order.get('order_value', 0) or 0 for order in reseller_orders)
            end_customer_count = len(reseller_data.get('end_customer_relationships', []))

            # Get top products sold through this reseller
            reseller_products = defaultdict(int)
            for order in reseller_data.get('orders', []):
                for product in order.get('products', []):
                    reseller_products[product.get('name', '')] += product.get('quantity', 1) or 1

            top_reseller_products = sorted(reseller_products.items(), key=lambda x: x[1], reverse=True)[:5]

            # Calculate industry distribution of end customers
            end_customer_industries = defaultdict(int)
            for end_customer_name in reseller_data.get('end_customer_relationships', []):
                # Find end customer in client_db
                for client_id, client_data in client_db.items():
                    if match_company_names(end_customer_name, client_data.get('company_name', '')):
                        industry = client_data.get('industry', 'Unknown')
                        end_customer_industries[industry] += 1
                        break

            # Create reseller insight
            reseller_insight = {
                'reseller_id': reseller_id,
                'reseller_name': reseller_name,
                'total_revenue': total_reseller_revenue,
                'end_customer_count': end_customer_count,
                'top_products': [{'name': p[0], 'quantity': p[1]} for p in top_reseller_products],
                'end_customer_industries': dict(end_customer_industries)
            }

            analysis_results['reseller_insights'].append(reseller_insight)

    # Sort reseller insights by revenue
    analysis_results['reseller_insights'].sort(key=lambda x: x.get('total_revenue', 0), reverse=True)

    # === PRODUCT INSIGHTS ===
    # Identify top-performing products
    sorted_products = sorted(product_db.items(), key=lambda x: x[1].get('total_revenue', 0), reverse=True)

    for product_id, product_data in sorted_products[:10]:  # Top 10 products
        success_factors = []

        # Analyze what makes this product successful
        if product_data.get('conversion_rate', 0) > 60:
            success_factors.append("High conversion rate")

        if len(product_data.get('clients', [])) > 5:
            success_factors.append("Broad customer appeal")

        if product_data.get('total_units_sold', 0) > 20:
            success_factors.append("High volume sales")

        if product_data.get('price_range', {}).get('max', 0) - product_data.get('price_range', {}).get('min', 0) > 50:
            success_factors.append("Flexible pricing strategy")

        # Compare direct vs reseller performance
        reseller_count = len(product_data.get('reseller_orders', []))
        direct_count = len(product_data.get('direct_orders', []))

        channel_mix = "Primarily direct sales"
        if reseller_count > direct_count * 2:
            channel_mix = "Primarily reseller sales"
        elif reseller_count > 0 and direct_count > 0:
            channel_mix = "Mixed sales channels"

        # Generate product insight
        product_insight = {
            'product_name': product_data.get('name'),
            'product_id': product_id,
            'category': product_data.get('category', 'Other'),
            'total_revenue': product_data.get('total_revenue', 0),
            'units_sold': product_data.get('total_units_sold', 0),
            'customer_count': len(product_data.get('clients', [])),
            'conversion_rate': product_data.get('conversion_rate', 0),
            'success_factors': success_factors,
            'channel_mix': channel_mix,
            'reseller_vs_direct': {
                'reseller_percentage': product_data.get('reseller_order_percentage', 0),
                'reseller_avg_price': product_data.get('reseller_average_price'),
                'direct_avg_price': product_data.get('direct_average_price')
            },
            'industry_distribution': product_data.get('industry_distribution', {})
        }

        analysis_results['product_insights'].append(product_insight)

    # === CROSS-SELL ANALYSIS ===
    # Find products frequently purchased together
    product_combinations = defaultdict(int)

    for client_id, client_data in client_db.items():
        client_products = [product.get('name') for product in client_data.get('products', [])]

        # Count product combinations
        for i in range(len(client_products)):
            for j in range(i + 1, len(client_products)):
                if client_products[i] and client_products[j]:
                    # Create a consistent key by sorting product names
                    combo_key = '-'.join(sorted([client_products[i], client_products[j]]))
                    product_combinations[combo_key] += 1

    # Find top combinations
    top_combos = sorted(product_combinations.items(), key=lambda x: x[1], reverse=True)[:10]

    for combo_key, frequency in top_combos:
        if frequency < 2:  # Only include if at least 2 clients bought this combination
            continue

        product_names = combo_key.split('-')

        # Find match in product database for more details
        product_ids = []
        for product_name in product_names:
            for product_id, product_data in product_db.items():
                if normalize_product_name(product_name) == normalize_product_name(product_data.get('name')):
                    product_ids.append(product_id)
                    break

        cross_sell = {
            'product_combination': product_names,
            'frequency': frequency,
            'product_ids': product_ids,
            'suggestion': f"Customers who purchase {product_names[0]} often also buy {product_names[1]}"
        }

        analysis_results['cross_sell_opportunities'].append(cross_sell)

    # === PRICING INSIGHTS ===
    # Analyze pricing patterns by product category
    category_pricing = defaultdict(list)

    for product_id, product_data in product_db.items():
        category = product_data.get('category', 'Other')
        avg_price = product_data.get('average_price')

        if avg_price:
            category_pricing[category].append({
                'product': product_data.get('name'),
                'price': avg_price,
                'conversion_rate': product_data.get('conversion_rate', 0),
                'product_id': product_id,
                'reseller_avg_price': product_data.get('reseller_average_price'),
                'direct_avg_price': product_data.get('direct_average_price')
            })

    # Find optimal price points by category
    for category, products in category_pricing.items():
        if len(products) < 2:  # Need at least 2 products in category for comparison
            continue

        # Sort by conversion rate
        high_converting = sorted(products, key=lambda x: x.get('conversion_rate', 0), reverse=True)[:3]

        # Calculate average price of high-converting products
        optimal_price = sum(p.get('price', 0) for p in high_converting) / len(high_converting)

        # Compare reseller vs direct pricing
        reseller_prices = [p.get('reseller_avg_price') for p in high_converting if p.get('reseller_avg_price')]
        direct_prices = [p.get('direct_avg_price') for p in high_converting if p.get('direct_avg_price')]

        reseller_optimal = sum(reseller_prices) / len(reseller_prices) if reseller_prices else None
        direct_optimal = sum(direct_prices) / len(direct_prices) if direct_prices else None

        pricing_insight = {
            'category': category,
            'optimal_price_point': optimal_price,
            'reseller_optimal_price': reseller_optimal,
            'direct_optimal_price': direct_optimal,
            'high_converting_examples': [p.get('product') for p in high_converting],
            'sample_conversion_rates': [p.get('conversion_rate', 0) for p in high_converting]
        }

        analysis_results['pricing_insights'].append(pricing_insight)

    # === INDUSTRY-PRODUCT MATRIX ===
    # Create matrix of industries and their preferred products
    industry_product_matrix = defaultdict(lambda: defaultdict(int))

    for client_id, client_data in client_db.items():
        industry = client_data.get('industry', 'Unknown')

        for product in client_data.get('products', []):
            product_name = product.get('name')
            if product_name:
                industry_product_matrix[industry][product_name] += 1

    # Convert to standard dict for JSON serialization
    for industry, products in industry_product_matrix.items():
        # Get top 3 products for each industry
        top_products = sorted(products.items(), key=lambda x: x[1], reverse=True)[:3]
        analysis_results['industry_product_matrix'][industry] = [
            {'product': p[0], 'frequency': p[1]} for p in top_products
        ]

    # === GENERAL INSIGHTS ===
    # Add high-level insights based on all the analysis

    # Insight 1: Most valuable client segment
    if analysis_results['client_segments']:
        top_segment = analysis_results['client_segments'][0]
        analysis_results['general_insights'].append(
            f"The {top_segment['industry']} industry ({top_segment['size']} businesses) represents our most valuable client segment with {top_segment['client_count']} clients and ${top_segment['total_spend']:.2f} in total spend."
        )

    # Insight 2: Best-selling product category
    category_revenues = defaultdict(float)
    for product_id, product_data in product_db.items():
        category = product_data.get('category', 'Other')
        category_revenues[category] += product_data.get('total_revenue', 0)

    if category_revenues:
        top_category = max(category_revenues.items(), key=lambda x: x[1])
        analysis_results['general_insights'].append(
            f"Our {top_category[0]} category is our best-selling product line, generating ${top_category[1]:.2f} in revenue."
        )

    # Insight 3: Optimal pricing strategy
    if analysis_results['pricing_insights']:
        pricing_insight = analysis_results['pricing_insights'][0]
        analysis_results['general_insights'].append(
            f"Products in the {pricing_insight['category']} category have the highest conversion rates at price points around ${pricing_insight['optimal_price_point']:.2f}."
        )

    # Insight 4: Cross-selling opportunity
    if analysis_results['cross_sell_opportunities']:
        cross_sell = analysis_results['cross_sell_opportunities'][0]
        analysis_results['general_insights'].append(
            f"There's a strong opportunity to cross-sell {cross_sell['product_combination'][0]} with {cross_sell['product_combination'][1]} as {cross_sell['frequency']} customers have purchased both."
        )

    # Insight 5: Reseller channel
    if analysis_results['reseller_insights']:
        top_reseller = analysis_results['reseller_insights'][0]
        analysis_results['general_insights'].append(
            f"{top_reseller['reseller_name']} is our most valuable reseller partner, generating ${top_reseller['total_revenue']:.2f} in revenue across {top_reseller['end_customer_count']} end customers."
        )

    print("Combined analysis complete.")
    logger.info("Combined analysis complete.")

    return analysis_results


# === VISUALIZATION FUNCTIONS ===
def generate_visualizations(client_db, product_db, reseller_relationships, combined_analysis):
    """Generate visualizations from the analysis data."""
    print("Generating visualizations...")

    # 1. Client industry distribution pie chart
    industry_counts = defaultdict(int)
    for client_id, client_data in client_db.items():
        # Only count end customers, not resellers
        if not client_data.get('is_reseller', False):
            industry_counts[client_data.get('industry', 'Unknown')] += 1

    plt.figure(figsize=(10, 6))
    plt.pie(industry_counts.values(), labels=industry_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('End Customer Distribution by Industry')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'client_industry_distribution.png'))
    plt.close()

    # 2. Top 10 products by revenue bar chart
    top_products = sorted(product_db.items(), key=lambda x: x[1].get('total_revenue', 0), reverse=True)[:10]
    product_names = [p[1].get('name', '')[:20] + '...' if len(p[1].get('name', '')) > 20 else p[1].get('name', '') for p
                     in top_products]
    revenues = [p[1].get('total_revenue', 0) for p in top_products]

    plt.figure(figsize=(12, 6))
    plt.bar(product_names, revenues)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 10 Products by Revenue')
    plt.ylabel('Revenue ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'top_products_revenue.png'))
    plt.close()

    # 3. Product category conversion rates
    categories = defaultdict(list)
    for product_id, product_data in product_db.items():
        category = product_data.get('category', 'Other')
        conversion = product_data.get('conversion_rate', 0)
        categories[category].append(conversion)

    avg_conversion = {cat: sum(rates) / len(rates) for cat, rates in categories.items() if rates}

    plt.figure(figsize=(10, 6))
    plt.bar(avg_conversion.keys(), avg_conversion.values())
    plt.title('Average Conversion Rate by Product Category')
    plt.ylabel('Conversion Rate (%)')
    plt.axhline(y=sum(avg_conversion.values()) / len(avg_conversion), color='r', linestyle='--',
                label='Overall Average')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'category_conversion_rates.png'))
    plt.close()

    # 4. Direct vs Reseller Sales Comparison
    direct_sales = sum(client_data.get('total_spend', 0) for client_id, client_data in client_db.items()
                       if not client_data.get('is_reseller', False) and not client_data.get('is_end_customer', False))

    reseller_sales = sum(rel_data.get('total_order_value', 0) for rel_key, rel_data in reseller_relationships.items())

    if direct_sales > 0 or reseller_sales > 0:
        plt.figure(figsize=(8, 8))
        plt.pie([direct_sales, reseller_sales],
                labels=['Direct Sales', 'Reseller Sales'],
                autopct='%1.1f%%',
                startangle=90,
                colors=['#ff9999', '#66b3ff'])
        plt.axis('equal')
        plt.title('Revenue Split: Direct vs Reseller Sales')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'direct_vs_reseller_sales.png'))
        plt.close()

    # 5. Top Resellers by Revenue
    if combined_analysis.get('reseller_insights'):
        top_resellers = combined_analysis['reseller_insights'][:5]  # Top 5 resellers
        reseller_names = [
            r.get('reseller_name', '')[:15] + '...' if len(r.get('reseller_name', '')) > 15 else r.get('reseller_name',
                                                                                                       '') for r in
            top_resellers]
        reseller_revenues = [r.get('total_revenue', 0) for r in top_resellers]

        plt.figure(figsize=(10, 6))
        plt.bar(reseller_names, reseller_revenues)
        plt.title('Top Resellers by Revenue')
        plt.ylabel('Revenue ($)')
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_FOLDER, 'top_resellers_revenue.png'))
        plt.close()

    print(f"Visualizations saved to {VISUALIZATION_FOLDER}")
    logger.info(f"Visualizations saved to {VISUALIZATION_FOLDER}")


# === MAIN EXECUTION ===
def main():
    print("\n==== ENHANCED MULTI-PARTY RELATIONSHIP CATEGORIZATION SYSTEM ====\n")

    # Load the detailed extraction and comprehensive analysis results
    print("Loading detailed extraction results...")
    detailed_results = load_json_file(DETAILED_EXTRACTION_FILE)

    if not detailed_results:
        print("❌ Error: No detailed extraction results found. Exiting.")
        return

    print(f"Loaded {len(detailed_results)} detailed extraction results.")

    print("Loading comprehensive analysis results...")
    comprehensive_results = load_json_file(COMPREHENSIVE_ANALYSIS_FILE)

    if not comprehensive_results:
        print("❌ Error: No comprehensive analysis results found. Exiting.")
        return

    print(f"Loaded {len(comprehensive_results)} comprehensive analysis results.")

    # Perform client categorization with enhanced reseller detection
    client_db = categorize_by_client(detailed_results, comprehensive_results)
    save_results(client_db, CLIENT_CATEGORIZATION_FILE)

    # Load the reseller relationships created during client categorization
    reseller_relationships = load_json_file(RESELLER_RELATIONSHIPS_FILE)
    print(f"Loaded {len(reseller_relationships)} reseller relationships.")

    # Perform product categorization with reseller order tracking
    product_db = categorize_by_product(detailed_results, comprehensive_results, client_db)
    save_results(product_db, PRODUCT_CATEGORIZATION_FILE)

    # Perform combined analysis with reseller insights
    combined_analysis = perform_combined_analysis(client_db, product_db, reseller_relationships)
    save_results(combined_analysis, COMBINED_ANALYSIS_FILE)

    # Generate visualizations with reseller comparisons
    generate_visualizations(client_db, product_db, reseller_relationships, combined_analysis)

    print("\n🎉 Enhanced categorization and analysis complete!")
    print(
        f"Identified {len(client_db)} unique clients, {len(product_db)} unique products, and {len(reseller_relationships)} reseller relationships.")
    print(f"Generated {len(combined_analysis['general_insights'])} high-level insights.")

    print(f"\nResults saved to:")
    print(f"- Client categorization: {CLIENT_CATEGORIZATION_FILE}")
    print(f"- Product categorization: {PRODUCT_CATEGORIZATION_FILE}")
    print(f"- Reseller relationships: {RESELLER_RELATIONSHIPS_FILE}")
    print(f"- Combined analysis: {COMBINED_ANALYSIS_FILE}")
    print(f"- Visualizations: {VISUALIZATION_FOLDER}")

    # Perform communication effectiveness analysis
    print("Analyzing communication effectiveness...")
    communication_insights = analyze_communication_effectiveness(comprehensive_results)
    save_results(communication_insights, os.path.join(VISUALIZATION_FOLDER, "communication_effectiveness.json"))

    # Print top communication insights
    print("\nTop communication patterns:")
    for i, insight in enumerate(communication_insights.get('top_patterns', [])):
        print(f"{i + 1}. {insight}")

    if communication_insights.get('winning_combination'):
        print(f"\nWinning email formula: {communication_insights['winning_combination']}")

    # Print sample insights about reseller relationships
    print("\nSample reseller insights:")
    if combined_analysis.get('reseller_insights'):
        for i, insight in enumerate(combined_analysis['reseller_insights'][:3]):
            print(
                f"{i + 1}. {insight['reseller_name']} has {insight['end_customer_count']} end customers and generated ${insight['total_revenue']:.2f} in revenue.")
    else:
        print("No reseller insights available.")

    print("\nSample general insights:")
    for i, insight in enumerate(combined_analysis['general_insights']):
        print(f"{i + 1}. {insight}")

    logger.info("Enhanced categorization and analysis completed successfully.")


if __name__ == "__main__":
    main()