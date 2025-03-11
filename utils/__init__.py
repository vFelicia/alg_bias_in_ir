# Make utils directory a proper Python package
# This allows imports like: from utils.preprocessing import preprocess_text

from utils.preprocessing import preprocess_text, visualize_preprocessing
from utils.indexing import InvertedIndex
from utils.tfidf import compute_tfidf, visualize_tfidf_comparison
from utils.retrieval import search_documents, boolean_search
from utils.visualization import plot_corpus_stats, visualize_cultural_term_comparison