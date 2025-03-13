# utils/indexing.py
import re
import os
from collections import defaultdict
import nltk
from nltk.stem import PorterStemmer
import pandas as pd
import streamlit as st
import json

# Import preprocessing utilities
from utils.preprocessing import preprocess_text

class InvertedIndex:
    """
    Inverted index implementation for educational purposes.
    Shows how documents are indexed and demonstrates bias entry points.
    """
    
    def __init__(self, preserve_case=False, use_stemming=True, include_positions=True, use_custom_dict=False):
        """
        Initialize the inverted index
        
        Parameters:
        -----------
        preserve_case : bool, default=False
            Whether to preserve case information
        use_stemming : bool, default=True
            Whether to apply stemming during indexing
        include_positions : bool, default=True
            Whether to store word positions in the index
        use_custom_dict : bool, default=False
            Use dictionary instead of defaultdict with lambda for pickling
        """
        self.preserve_case = preserve_case
        self.use_stemming = use_stemming
        self.include_positions = include_positions
        
        # Main inverted index: {term: {doc_id: [positions]}}
        if use_custom_dict:
            # Use regular dict for pickling compatibility
            self.index = {}
        else:
            # Use defaultdict with lambda (not picklable)
            self.index = defaultdict(lambda: defaultdict(list))
        
        # Document lengths (for normalization)
        self.doc_lengths = {}
        
        # Document store
        self.documents = {}
        
        # Document metadata
        self.metadata = {}
        
        # Preprocessing steps visibility (for educational purposes)
        self.preprocessing_steps = {}
    
    def add_document(self, doc_id, text, metadata=None):
        """
        Add a document to the index
        
        Parameters:
        -----------
        doc_id : str or int
            Unique document identifier
        text : str
            Document text
        metadata : dict, optional
            Document metadata
            
        Returns:
        --------
        dict
            Preprocessing steps for educational visualization
        """
        # Store original document
        self.documents[doc_id] = text
        
        # Store metadata if provided
        if metadata:
            self.metadata[doc_id] = metadata
        
        # Preprocess text
        preprocessing_result = preprocess_text(
            text, 
            remove_stopwords=True, 
            do_stemming=self.use_stemming,
            keep_case=self.preserve_case
        )
        
        # Store preprocessing steps for visualization
        self.preprocessing_steps[doc_id] = preprocessing_result
        
        # Get processed tokens
        tokens = preprocessing_result['stemmed'] if self.use_stemming else preprocessing_result['tokens']
        
        # Calculate document length (for cosine similarity later)
        self.doc_lengths[doc_id] = len(tokens)
        
        # Add to index with positions
        if isinstance(self.index, defaultdict):
            # Original code for defaultdict
            if self.include_positions:
                for position, token in enumerate(tokens):
                    self.index[token][doc_id].append(position)
            else:
                # Just add document to term's posting list without positions
                for token in set(tokens):  # Use set to avoid duplicates
                    count = tokens.count(token)
                    self.index[token][doc_id] = count
        else:
            # Code for regular dict
            if self.include_positions:
                for position, token in enumerate(tokens):
                    if token not in self.index:
                        self.index[token] = {}
                    if doc_id not in self.index[token]:
                        self.index[token][doc_id] = []
                    self.index[token][doc_id].append(position)
            else:
                for token in set(tokens):
                    if token not in self.index:
                        self.index[token] = {}
                    count = tokens.count(token)
                    self.index[token][doc_id] = count
        
        # Store metadata if provided
        if metadata:
            self.metadata[doc_id] = metadata
        
        return preprocessing_result
    
    def add_directory(self, directory_path, extension='.txt', metadata_file=None):
        """
        Index all documents in a directory
        
        Parameters:
        -----------
        directory_path : str
            Path to directory containing documents
        extension : str, default='.txt'
            File extension to look for
        metadata_file : str, optional
            Path to metadata CSV file with doc_id column
            
        Returns:
        --------
        int
            Number of documents indexed
        """
        # Load metadata if provided
        metadata_dict = {}
        if metadata_file and os.path.exists(metadata_file):
            metadata_df = pd.read_csv(metadata_file)
            if 'doc_id' in metadata_df.columns:
                for _, row in metadata_df.iterrows():
                    doc_id = row['doc_id']
                    metadata_dict[doc_id] = row.to_dict()
        
        # Index all documents in directory
        count = 0
        for filename in os.listdir(directory_path):
            if filename.endswith(extension):
                doc_id = filename.replace(extension, '')
                file_path = os.path.join(directory_path, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Add document with metadata if available
                doc_metadata = metadata_dict.get(doc_id, None)
                self.add_document(doc_id, text, doc_metadata)
                count += 1
        
        return count
    
    def search(self, query, use_same_preprocessing=True, top_k=10):
        """
        Search the index for a query
        
        Parameters:
        -----------
        query : str
            Search query
        use_same_preprocessing : bool, default=True
            Whether to use the same preprocessing as when indexing
        top_k : int, default=10
            Number of top results to return
            
        Returns:
        --------
        list
            List of (doc_id, score) tuples
        dict
            Search process visualization data
        """
        # Preprocess query
        if use_same_preprocessing:
            query_result = preprocess_text(
                query,
                remove_stopwords=True,
                do_stemming=self.use_stemming,
                keep_case=self.preserve_case
            )
            query_tokens = query_result['stemmed'] if self.use_stemming else query_result['tokens']
        else:
            # Simple tokenization with no preprocessing
            query_tokens = query.lower().split()
        
        # For visualization purposes
        search_process = {
            'query': query,
            'processed_query': query_tokens,
            'token_postings': {},
            'scoring_details': {}
        }
        
        # Simple TF-IDF scoring
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token in self.index:
                # Get document frequency
                df = len(self.index[token])
                # Simple IDF calculation
                idf = len(self.documents) / df if df > 0 else 0
                
                # Store token postings for visualization
                search_process['token_postings'][token] = {
                    'df': df,
                    'idf': idf,
                    'docs': list(self.index[token].keys())
                }
                
                # Score each document containing the term
                for doc_id, positions in self.index[token].items():
                    # Simple TF calculation (normalized by doc length)
                    tf = len(positions) / self.doc_lengths[doc_id] if self.include_positions else positions / self.doc_lengths[doc_id]
                    
                    # TF-IDF score
                    tfidf = tf * idf
                    scores[doc_id] += tfidf
                    
                    # Store scoring details for visualization
                    if doc_id not in search_process['scoring_details']:
                        search_process['scoring_details'][doc_id] = {}
                    
                    search_process['scoring_details'][doc_id][token] = {
                        'tf': tf,
                        'positions': positions if self.include_positions else None,
                        'contribution': tfidf
                    }
        
        # Sort by score and return top k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Add final scores to search process
        search_process['final_scores'] = {doc_id: score for doc_id, score in sorted_results}
        
        return sorted_results, search_process
    
    def phrase_search(self, phrase, top_k=10):
        """
        Search for an exact phrase in the index (requires positional index)
        
        Parameters:
        -----------
        phrase : str
            Phrase to search for
        top_k : int, default=10
            Number of top results to return
            
        Returns:
        --------
        list
            List of (doc_id, score) tuples
        dict
            Search process visualization data
        """
        if not self.include_positions:
            return [], {"error": "Phrase search requires positional indexing"}
        
        # Preprocess phrase
        phrase_result = preprocess_text(
            phrase,
            remove_stopwords=False,  # Keep all words for phrase search
            do_stemming=self.use_stemming,
            keep_case=self.preserve_case
        )
        phrase_tokens = phrase_result['stemmed'] if self.use_stemming else phrase_result['tokens']
        
        # For visualization purposes
        search_process = {
            'phrase': phrase,
            'processed_phrase': phrase_tokens,
            'matching_details': {}
        }
        
        # Find documents containing the first term
        if not phrase_tokens or phrase_tokens[0] not in self.index:
            return [], search_process
        
        candidates = self.index[phrase_tokens[0]]
        
        # Check if all terms exist in the index
        for token in phrase_tokens[1:]:
            if token not in self.index:
                return [], search_process
        
        # Check for phrase matches
        matches = {}
        
        for doc_id, positions in candidates.items():
            # Check if document contains all other terms
            valid_doc = True
            for token in phrase_tokens[1:]:
                if doc_id not in self.index[token]:
                    valid_doc = False
                    break
            
            if not valid_doc:
                continue
            
            # Check for consecutive positions
            match_positions = []
            for pos in positions:
                match = True
                for i, token in enumerate(phrase_tokens[1:], 1):
                    if pos + i not in self.index[token][doc_id]:
                        match = False
                        break
                
                if match:
                    match_positions.append(pos)
            
            # Store matches for scoring
            if match_positions:
                matches[doc_id] = match_positions
                
                # Store matching details for visualization
                search_process['matching_details'][doc_id] = {
                    'match_positions': match_positions,
                    'context': self._get_match_context(doc_id, match_positions[0], len(phrase_tokens))
                }
        
        # Score matches (by number of occurrences)
        scored_matches = [(doc_id, len(positions)) for doc_id, positions in matches.items()]
        sorted_results = sorted(scored_matches, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Add final scores to search process
        search_process['final_scores'] = {doc_id: score for doc_id, score in sorted_results}
        
        return sorted_results, search_process
    
    def _get_match_context(self, doc_id, position, phrase_length, context_size=5):
        """
        Get context around a phrase match for visualization
        
        Parameters:
        -----------
        doc_id : str or int
            Document identifier
        position : int
            Starting position of the match
        phrase_length : int
            Length of the phrase in tokens
        context_size : int, default=5
            Number of tokens to show before and after the match
            
        Returns:
        --------
        dict
            Context information
        """
        # Get the original document
        original_text = self.documents[doc_id]
        
        # Get the tokens
        tokens = self.preprocessing_steps[doc_id]['tokens']
        
        # Determine context range
        start_pos = max(0, position - context_size)
        end_pos = min(len(tokens), position + phrase_length + context_size)
        
        # Get context tokens
        before_tokens = tokens[start_pos:position]
        match_tokens = tokens[position:position + phrase_length]
        after_tokens = tokens[position + phrase_length:end_pos]
        
        return {
            'before': ' '.join(before_tokens),
            'match': ' '.join(match_tokens),
            'after': ' '.join(after_tokens),
            'position': position,
            'start_pos': start_pos,
            'end_pos': end_pos
        }
    
    def visualize_index(self, max_terms=20):
        """
        Create a visualization of the inverted index for educational purposes
        
        Parameters:
        -----------
        max_terms : int, default=20
            Maximum number of terms to show
            
        Returns:
        --------
        dict
            Index visualization data
        """
        # Get most common terms
        term_doc_counts = [(term, len(postings)) for term, postings in self.index.items()]
        sorted_terms = sorted(term_doc_counts, key=lambda x: x[1], reverse=True)[:max_terms]
        
        # Prepare visualization data
        visualization_data = {
            'index_stats': {
                'total_terms': len(self.index),
                'total_documents': len(self.documents),
                'average_terms_per_doc': sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0
            },
            'common_terms': [],
            'term_details': {}
        }
        
        # Add common terms data
        for term, doc_count in sorted_terms:
            term_data = {
                'term': term,
                'document_count': doc_count,
                'document_percentage': (doc_count / len(self.documents)) * 100 if self.documents else 0
            }
            
            visualization_data['common_terms'].append(term_data)
            
            # Add detailed posting list for top terms
            posting_list = self.index[term]
            term_details = {
                'document_count': doc_count,
                'posting_list': {}
            }
            
            # Add sample of posting list (first 5 documents)
            for i, (doc_id, positions) in enumerate(posting_list.items()):
                if i >= 5:  # Limit to 5 documents for readability
                    break
                
                if self.include_positions:
                    term_details['posting_list'][doc_id] = {
                        'count': len(positions),
                        'positions': positions[:10] + ['...'] if len(positions) > 10 else positions  # Show first 10 positions
                    }
                else:
                    term_details['posting_list'][doc_id] = {
                        'count': positions
                    }
            
            visualization_data['term_details'][term] = term_details
        
        return visualization_data
    
    def bias_analysis(self):
        """
        Analyze potential bias in the index
        
        Returns:
        --------
        dict
            Bias analysis data
        """
        analysis = {
            'stemming_effects': {},
            'case_sensitivity': {},
            'document_representation': {}
        }
        
        # Analyze stemming effects (if enabled)
        if self.use_stemming:
            # Find terms that map to the same stem
            stem_to_terms = defaultdict(list)
            
            # For educational purposes, re-stem a sample of original terms
            stemmer = PorterStemmer()
            
            # Collect original terms from documents
            all_original_terms = set()
            for doc_id, text in self.documents.items():
                # Simple tokenization to get original terms
                tokens = re.findall(r'\b\w+\b', text.lower())
                all_original_terms.update(tokens)
            
            # Map stems to original terms
            for term in all_original_terms:
                stem = stemmer.stem(term)
                stem_to_terms[stem].append(term)
            
            # Filter to stems with multiple original terms
            for stem, terms in stem_to_terms.items():
                if len(terms) > 1:
                    if stem in self.index:
                        doc_count = len(self.index[stem])
                        analysis['stemming_effects'][stem] = {
                            'original_terms': terms,
                            'document_count': doc_count,
                            'potential_bias': "High" if doc_count > len(self.documents) / 2 else "Medium" if doc_count > len(self.documents) / 4 else "Low"
                        }
        
        # Analyze case sensitivity effects
        if not self.preserve_case:
            # For educational purposes, sample some original texts to show case loss
            case_examples = {}
            
            for doc_id, text in list(self.documents.items())[:5]:  # Use first 5 docs as examples
                # Find terms with capital letters
                capitalized_tokens = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
                
                if capitalized_tokens:
                    # Show how these were indexed (all lowercase)
                    examples = {}
                    for token in capitalized_tokens[:5]:  # Show first 5 examples
                        lower_token = token.lower()
                        if lower_token in self.index:
                            examples[token] = {
                                'indexed_as': lower_token,
                                'in_documents': len(self.index[lower_token])
                            }
                    
                    if examples:
                        case_examples[doc_id] = examples
            
            analysis['case_sensitivity'] = {
                'potential_bias': "Names and proper nouns lose distinction",
                'examples': case_examples
            }
        
        # Analyze document representation
        doc_term_counts = {doc_id: len(set(preprocessing['tokens'])) for doc_id, preprocessing in self.preprocessing_steps.items()}
        avg_terms = sum(doc_term_counts.values()) / len(doc_term_counts) if doc_term_counts else 0
        
        # Find documents with significantly different term counts
        outliers = {}
        for doc_id, term_count in doc_term_counts.items():
            if term_count < avg_terms / 2 or term_count > avg_terms * 2:
                outliers[doc_id] = {
                    'term_count': term_count,
                    'difference_from_avg': (term_count / avg_terms - 1) * 100,
                    'potential_bias': "Under-represented" if term_count < avg_terms / 2 else "Over-represented"
                }
        
        analysis['document_representation'] = {
            'average_unique_terms': avg_terms,
            'outlier_documents': outliers
        }
        
        return analysis
    
    def get_document_snippet(self, doc_id, max_length=200):
        """
        Get a snippet of a document for display
        
        Parameters:
        -----------
        doc_id : str or int
            Document identifier
        max_length : int, default=200
            Maximum length of the snippet
            
        Returns:
        --------
        str
            Document snippet
        """
        if doc_id not in self.documents:
            return ""
        
        text = self.documents[doc_id]
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "..."
    
    def streamlit_index_explorer(self):
        """
        Interactive Streamlit widget for exploring the index
        """
        st.subheader("Inverted Index Explorer")
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documents", len(self.documents))
        
        with col2:
            st.metric("Unique Terms", len(self.index))
        
        with col3:
            avg_terms = sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0
            st.metric("Avg Terms per Document", f"{avg_terms:.1f}")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Term Explorer", "Document Explorer", "Bias Analysis"])
        
        with tab1:
            # Term search
            term_search = st.text_input("Search for a term in the index:")
            
            if term_search:
                term_search = term_search.lower()
                if term_search in self.index:
                    # Show term stats
                    st.write(f"### Term: '{term_search}'")
                    
                    doc_count = len(self.index[term_search])
                    st.write(f"**Found in {doc_count} documents** ({(doc_count / len(self.documents)) * 100:.1f}% of corpus)")
                    
                    # Show posting list
                    st.write("### Posting List (first 10 documents)")
                    
                    for i, (doc_id, positions) in enumerate(self.index[term_search].items()):
                        if i >= 10:  # Limit to 10 documents
                            st.write("...")
                            break
                        
                        if self.include_positions:
                            positions_str = str(positions[:5])
                            if len(positions) > 5:
                                positions_str = positions_str[:-1] + ", ...]"
                            st.write(f"**Document '{doc_id}'**: {len(positions)} occurrences at positions {positions_str}")
                        else:
                            st.write(f"**Document '{doc_id}'**: {positions} occurrences")
                        
                        # Show snippet
                        st.write(f"Snippet: _{self.get_document_snippet(doc_id)}_")
                else:
                    st.warning(f"Term '{term_search}' not found in the index.")
                    
                    # Suggest similar terms if any
                    similar_terms = [term for term in self.index.keys() if term_search in term]
                    if similar_terms:
                        st.write("Did you mean one of these terms?")
                        for term in similar_terms[:5]:  # Show up to 5 suggestions
                            st.write(f"- {term} (in {len(self.index[term])} documents)")
            else:
                # Show most common terms
                st.write("### Most Common Terms")
                
                term_data = self.visualize_index(max_terms=20)
                
                # Create dataframe for display
                common_terms_df = pd.DataFrame(term_data['common_terms'])
                st.dataframe(common_terms_df)
        
        with tab2:
            # Document explorer
            doc_ids = list(self.documents.keys())
            selected_doc = st.selectbox("Select a document to explore:", doc_ids)
            
            if selected_doc:
                st.write(f"### Document: '{selected_doc}'")
                
                # Show document metadata if available
                if selected_doc in self.metadata:
                    st.write("#### Metadata")
                    st.json(self.metadata[selected_doc])
                
                # Show document statistics
                st.write("#### Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Original Length", len(self.documents[selected_doc]))
                    
                with col2:
                    st.metric("Indexed Terms", self.doc_lengths[selected_doc])
                
                # Show preprocessing steps
                if st.checkbox("Show Preprocessing Steps"):
                    st.write("#### Preprocessing Steps")
                    
                    preproc = self.preprocessing_steps[selected_doc]
                    
                    # Original text
                    st.write("**Original Text:**")
                    st.text(preproc['original'][:500] + "..." if len(preproc['original']) > 500 else preproc['original'])
                    
                    # After lowercase
                    if 'lowercase' in preproc:
                        st.write("**After Lowercase:**")
                        st.text(preproc['lowercase'][:500] + "..." if len(preproc['lowercase']) > 500 else preproc['lowercase'])
                    
                    # After punctuation removal
                    if 'no_punctuation' in preproc:
                        st.write("**After Punctuation Removal:**")
                        st.text(preproc['no_punctuation'][:500] + "..." if len(preproc['no_punctuation']) > 500 else preproc['no_punctuation'])
                    
                    # After tokenization
                    if 'tokens' in preproc:
                        st.write("**After Tokenization:**")
                        st.text(str(preproc['tokens'][:50]) + "..." if len(preproc['tokens']) > 50 else str(preproc['tokens']))
                    
                    # After stopword removal
                    if 'no_stopwords' in preproc:
                        st.write("**After Stopword Removal:**")
                        st.text(str(preproc['no_stopwords'][:50]) + "..." if len(preproc['no_stopwords']) > 50 else str(preproc['no_stopwords']))
                    
                    # After stemming
                    if 'stemmed' in preproc:
                        st.write("**After Stemming:**")
                        st.text(str(preproc['stemmed'][:50]) + "..." if len(preproc['stemmed']) > 50 else str(preproc['stemmed']))
                
                # Sample of terms in this document
                st.write("#### Sample Terms in this Document")
                
                # Count terms in this document
                doc_terms = {}
                for term, postings in self.index.items():
                    if selected_doc in postings:
                        if self.include_positions:
                            doc_terms[term] = len(postings[selected_doc])
                        else:
                            doc_terms[term] = postings[selected_doc]
                
                # Sort by frequency
                sorted_terms = sorted(doc_terms.items(), key=lambda x: x[1], reverse=True)
                
                # Create dataframe for display
                terms_df = pd.DataFrame(sorted_terms[:20], columns=['Term', 'Occurrences'])
                st.dataframe(terms_df)
        
        with tab3:
            st.write("### Bias Analysis")
            
            bias_data = self.bias_analysis()
            
            # Stemming effects
            st.write("#### Stemming Effects")
            
            if bias_data['stemming_effects']:
                st.write("""
                Stemming reduces words to their root form, which can improve recall but may introduce bias by:
                - Treating different concepts as the same term
                - Affecting culturally specific terms differently than common English words
                """)
                
                # Show examples
                stemming_df = []
                for stem, data in list(bias_data['stemming_effects'].items())[:10]:  # Show first 10
                    stemming_df.append({
                        'Stem': stem,
                        'Original Terms': ', '.join(data['original_terms']),
                        'Document Count': data['document_count'],
                        'Potential Bias': data['potential_bias']
                    })
                
                st.dataframe(pd.DataFrame(stemming_df))
            else:
                st.info("No stemming bias detected or stemming is disabled.")
            
            # Case sensitivity
            st.write("#### Case Sensitivity Effects")
            
            if 'examples' in bias_data['case_sensitivity'] and bias_data['case_sensitivity']['examples']:
                st.write("""
                Case folding (converting all text to lowercase) can introduce bias by:
                - Losing distinction between proper nouns and common words
                - Affecting names from different cultures differently
                """)
                
                # Show examples
                case_df = []
                for doc_id, examples in bias_data['case_sensitivity']['examples'].items():
                    for original, data in examples.items():
                        case_df.append({
                            'Original Term': original,
                            'Indexed As': data['indexed_as'],
                            'In Documents': data['in_documents'],
                            'Document ID': doc_id
                        })
                
                st.dataframe(pd.DataFrame(case_df))
            else:
                st.info("No case sensitivity bias detected or case is preserved.")
            
            # Document representation
            st.write("#### Document Representation")
            
            if bias_data['document_representation']['outlier_documents']:
                st.write(f"""
                Documents with significantly different numbers of terms may be under or over-represented in search results.
                Average unique terms per document: {bias_data['document_representation']['average_unique_terms']:.1f}
                """)
                
                # Show outliers
                outlier_df = []
                for doc_id, data in bias_data['document_representation']['outlier_documents'].items():
                    outlier_df.append({
                        'Document ID': doc_id,
                        'Term Count': data['term_count'],
                        'Difference from Average': f"{data['difference_from_avg']:.1f}%",
                        'Potential Bias': data['potential_bias']
                    })
                
                st.dataframe(pd.DataFrame(outlier_df))
            else:
                st.info("No significant document representation bias detected.")

# Streamlit component for exploring inverted index
def streamlit_inverted_index_demo():
    """
    Create a Streamlit demonstration of the inverted index
    """
    st.title("Inverted Index: The Heart of Search Engines")
    
    st.markdown("""
    An inverted index is a data structure used by search engines to quickly find documents containing specific terms.
    The term "inverted" refers to the fact that it maps from terms to documents (rather than from documents to terms).
    
    Let's explore how this works and where bias can enter the system.
    """)
    
    # Sample texts
    sample_texts = {
        "doc1": "The quick brown fox jumps over the lazy dog. The dog was not amused.",
        "doc2": "A fox is usually red or brown in color. Foxes are cunning animals.",
        "doc3": "Laziness is often confused with efficiency. Work smarter, not harder.",
        "doc4": "The African American writer shared stories about her community's struggles and triumphs.",
        "doc5": "The Latinx student organization celebrated Hispanic Heritage Month with music and food."
    }
    
    # Create index with different settings
    preserve_case = st.checkbox("Preserve case", value=False)
    use_stemming = st.checkbox("Use stemming", value=True)
    include_positions = st.checkbox("Store word positions", value=True)
    
    index = InvertedIndex(preserve_case, use_stemming, include_positions)
    
    # Add sample documents
    for doc_id, text in sample_texts.items():
        index.add_document(doc_id, text)
    
    # Display the inverted index explorer
    index.streamlit_index_explorer()
    
    # Search interface
    st.subheader("Search the Index")
    
    search_type = st.radio("Search type:", ["Standard Search", "Phrase Search"])
    
    query = st.text_input("Enter your search query:")
    
    if query:
        if search_type == "Standard Search":
            results, process = index.search(query)
        else:
            results, process = index.phrase_search(query)
        
        # Display results
        st.write(f"### Search Results for: '{query}'")
        
        if results:
            for i, (doc_id, score) in enumerate(results, 1):
                st.write(f"**{i}. Document '{doc_id}'** (Score: {score:.4f})")
                st.write(f"> {index.get_document_snippet(doc_id)}")
        else:
            # Search interface section - completing where the code cuts off
            st.write("No results found for your query.")
        
        # Show search process visualization
        if st.checkbox("Show Search Process Details"):
            st.write("### Search Process Visualization")
            
            if search_type == "Standard Search":
                # Show query processing
                st.write("#### Query Processing")
                st.write(f"Original query: '{process['query']}'")
                st.write(f"Processed query tokens: {process['processed_query']}")
                
                # Show token postings
                st.write("#### Token Postings")
                for token, data in process['token_postings'].items():
                    st.write(f"**Term: '{token}'**")
                    st.write(f"- Document frequency: {data['df']} documents")
                    st.write(f"- IDF value: {data['idf']:.4f}")
                    st.write(f"- Found in documents: {', '.join(str(d) for d in data['docs'][:5])}" + ("..." if len(data['docs']) > 5 else ""))
                
                # Show scoring details
                st.write("#### Scoring Details")
                for doc_id, tokens in process['scoring_details'].items():
                    if doc_id in [doc_id for doc_id, _ in results]:  # Only show details for returned results
                        st.write(f"**Document '{doc_id}'**")
                        
                        # Create a score breakdown table
                        score_data = []
                        for token, details in tokens.items():
                            score_data.append({
                                'Term': token,
                                'TF': f"{details['tf']:.4f}",
                                'Occurrences': len(details['positions']) if self.include_positions else details['positions'],
                                'Score Contribution': f"{details['contribution']:.4f}"
                            })
                        
                        # Show as dataframe
                        if score_data:
                            st.dataframe(pd.DataFrame(score_data))
            else:
                # Phrase search process visualization
                st.write("#### Phrase Processing")
                st.write(f"Original phrase: '{process['phrase']}'")
                st.write(f"Processed phrase tokens: {process['processed_phrase']}")
                
                # Show matching details
                if process['matching_details']:
                    st.write("#### Matching Details")
                    for doc_id, details in process['matching_details'].items():
                        st.write(f"**Document '{doc_id}'**")
                        st.write(f"- Matches found at positions: {details['match_positions'][:5]}" + ("..." if len(details['match_positions']) > 5 else ""))
                        
                        # Show context around match
                        context = details['context']
                        st.write("- Context around first match:")
                        st.write(f"  ...{context['before']} **{context['match']}** {context['after']}...")
                else:
                    st.write("No matches found for this phrase query.")

if __name__ == "__main__":
    # This allows the file to be run directly as a Streamlit app
    streamlit_inverted_index_demo()