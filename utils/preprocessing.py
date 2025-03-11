# utils/preprocessing.py
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text, remove_stopwords=True, do_stemming=True, keep_case=False):
    """
    Preprocess text by applying various transformations.
    
    Parameters:
    -----------
    text : str
        The input text to preprocess
    remove_stopwords : bool, default=True
        Whether to remove stopwords
    do_stemming : bool, default=True
        Whether to apply stemming
    keep_case : bool, default=False
        Whether to preserve case information
        
    Returns:
    --------
    dict
        A dictionary containing the text at different preprocessing stages
    """
    # Initialize preprocessing stages dictionary
    stages = {
        'original': text,
    }
    
    # Step 1: Lowercase conversion (optional)
    if not keep_case:
        text = text.lower()
        stages['lowercase'] = text
    
    # Step 2: Punctuation removal
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    stages['no_punctuation'] = text_no_punct
    
    # Step 3: Tokenization
    tokens = word_tokenize(text_no_punct)
    stages['tokens'] = tokens
    
    # Step 4: Stopword removal (optional)
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        stages['no_stopwords'] = filtered_tokens
    else:
        filtered_tokens = tokens
    
    # Step 5: Stemming (optional)
    if do_stemming:
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        stages['stemmed'] = stemmed_tokens
        stages['processed_text'] = ' '.join(stemmed_tokens)
    else:
        stages['processed_text'] = ' '.join(filtered_tokens)
    
    return stages

def visualize_preprocessing(text, remove_stopwords=True, do_stemming=True, keep_case=False):
    """
    Process text and return a visual representation of each step for educational purposes.
    
    Parameters:
    -----------
    text : str
        The input text to preprocess
    remove_stopwords : bool, default=True
        Whether to remove stopwords
    do_stemming : bool, default=True
        Whether to apply stemming
    keep_case : bool, default=False
        Whether to preserve case information
        
    Returns:
    --------
    list of tuples
        A list of (step_name, description, input, output) tuples
    """
    steps = []
    
    # Apply preprocessing and capture steps
    stages = preprocess_text(text, remove_stopwords, do_stemming, keep_case)
    
    # Step 1: Lowercase conversion
    if not keep_case:
        steps.append((
            "Lowercase Conversion",
            "Convert all text to lowercase to standardize words regardless of capitalization.",
            stages['original'],
            stages['lowercase']
        ))
    
    # Step 2: Punctuation removal
    punct_input = stages['lowercase'] if not keep_case else stages['original']
    steps.append((
        "Punctuation Removal",
        "Remove punctuation marks that might interfere with token matching.",
        punct_input,
        stages['no_punctuation']
    ))
    
    # Step 3: Tokenization
    steps.append((
        "Tokenization",
        "Split text into individual words (tokens).",
        stages['no_punctuation'],
        str(stages['tokens'])
    ))
    
    # Step 4: Stopword removal
    if remove_stopwords:
        steps.append((
            "Stopword Removal",
            "Remove common words (like 'the', 'and', 'is') that typically don't carry much meaning for search.",
            str(stages['tokens']),
            str(stages['no_stopwords'])
        ))
    
    # Step 5: Stemming
    if do_stemming:
        stem_input = str(stages['no_stopwords']) if remove_stopwords else str(stages['tokens'])
        steps.append((
            "Stemming",
            "Reduce words to their root form to match similar words (e.g., 'running' → 'run').",
            stem_input,
            str(stages['stemmed'])
        ))
    
    # Final processed text
    steps.append((
        "Final Processed Text",
        "The text as it would be indexed in the search system.",
        text,
        stages['processed_text']
    ))
    
    return steps

def compare_cultural_terms_preprocessing(terms_list):
    """
    Compare how different types of terms are affected by preprocessing.
    
    Parameters:
    -----------
    terms_list : list of dict
        List of dictionaries with 'term' and 'category' keys
        
    Returns:
    --------
    dict
        A dictionary with term categories as keys and lists of 
        (original_term, processed_term) tuples as values
    """
    comparison = {}
    
    for term_info in terms_list:
        term = term_info['term']
        category = term_info['category']
        
        # Process the term with standard settings
        processed = preprocess_text(term)
        processed_term = processed['processed_text']
        
        # Add to comparison dictionary
        if category not in comparison:
            comparison[category] = []
        
        comparison[category].append((term, processed_term))
    
    return comparison