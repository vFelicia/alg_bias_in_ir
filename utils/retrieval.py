# utils/retrieval.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
import re

# Import our custom modules
from utils.preprocessing import preprocess_text
from utils.indexing import InvertedIndex

def search_documents(query, index=None, documents=None, num_results=5, search_type="standard"):
    """
    Search for documents matching a query
    
    Parameters:
    -----------
    query : str
        Search query
    index : InvertedIndex, optional
        Inverted index to use for search (required unless documents is provided)
    documents : dict, optional
        Documents to search (required if index is not provided)
    num_results : int, default=5
        Number of results to return
    search_type : str, default="standard"
        Type of search to perform ("standard", "phrase", "boolean")
        
    Returns:
    --------
    list
        List of (doc_id, score) tuples representing search results
    dict
        Search process visualization data
    """
    # If an index is provided, use it
    if index is not None:
        if search_type == "phrase":
            return index.phrase_search(query, top_k=num_results)
        else:  # standard search by default
            return index.search(query, top_k=num_results)
    
    # If no index but documents are provided, perform a basic vector space search
    elif documents is not None:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        
        # Create a corpus from the documents
        corpus = list(documents.values())
        doc_ids = list(documents.keys())
        
        # Add the query to the corpus
        all_texts = corpus + [query]
        
        # Calculate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Get the query vector (last row)
        query_vector = tfidf_matrix[-1]
        
        # Calculate similarity between query and each document
        similarities = []
        for i in range(len(corpus)):
            doc_vector = tfidf_matrix[i]
            similarity = cosine_similarity(query_vector, doc_vector)[0][0]
            similarities.append((doc_ids[i], similarity))
        
        # Sort by similarity (descending)
        sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)[:num_results]
        
        # Create a simple search process visualization
        search_process = {
            'query': query,
            'vectorizer': 'TF-IDF',
            'similarity_measure': 'Cosine similarity',
            'num_documents': len(corpus),
            'results': sorted_results
        }
        
        return sorted_results, search_process
    
    # Neither index nor documents provided
    else:
        return [], {'error': 'No index or documents provided'}

def boolean_search(query, index, use_stemming=True):
    """
    Perform a boolean search (AND, OR, NOT operations)
    
    Parameters:
    -----------
    query : str
        Boolean query string (e.g., "fox AND dog", "fox OR (dog AND NOT cat)")
    index : InvertedIndex
        Inverted index to search
    use_stemming : bool, default=True
        Whether to apply stemming to query terms
        
    Returns:
    --------
    list
        List of document IDs matching the query
    dict
        Search process visualization data
    """
    # Tokenize the query first
    tokens = re.findall(r'\(|\)|AND|OR|NOT|[a-zA-Z]+', query)
    tokens = [token.strip() for token in tokens if token.strip()]
    
    # Convert terms to their stemmed form if necessary
    processed_tokens = []
    term_mapping = {}  # Original to processed mapping for visualization
    
    for token in tokens:
        if token in ('AND', 'OR', 'NOT', '(', ')'):
            processed_tokens.append(token)
        else:
            if use_stemming:
                # Apply preprocessing but only extract the stemmed term
                processed = preprocess_text(token, do_stemming=True)
                processed_term = processed['stemmed'][0] if processed['stemmed'] else token.lower()
                processed_tokens.append(processed_term)
                term_mapping[token] = processed_term
            else:
                processed_tokens.append(token.lower())
                term_mapping[token] = token.lower()
    
    # For visualization purposes
    search_process = {
        'query': query,
        'tokens': tokens,
        'processed_tokens': processed_tokens,
        'term_mapping': term_mapping,
        'steps': []
    }
    
    # Parse and evaluate the boolean expression
    def parse_expression(tokens, pos=0):
        results = set()
        current_op = 'OR'  # Default operation is OR
        negate_next = False
        
        while pos < len(tokens):
            token = tokens[pos]
            pos += 1
            
            # Handle parentheses for subexpressions
            if token == '(':
                subresults, new_pos = parse_expression(tokens, pos)
                pos = new_pos
                
                if negate_next:
                    # NOT operation: all documents except those in subresults
                    all_docs = set(index.documents.keys())
                    subresults = all_docs - subresults
                    negate_next = False
                
                # Apply the current operation
                if current_op == 'AND':
                    results &= subresults
                else:  # OR
                    results |= subresults
                
                # Add step for visualization
                search_process['steps'].append({
                    'operation': f"Applied {current_op} with subexpression",
                    'result_count': len(results)
                })
            
            elif token == ')':
                return results, pos
            
            elif token == 'AND':
                current_op = 'AND'
            
            elif token == 'OR':
                current_op = 'OR'
            
            elif token == 'NOT':
                negate_next = True
            
            else:
                # This is a search term
                if token in index.index:
                    term_results = set(index.index[token].keys())
                    
                    # Apply negation if needed
                    if negate_next:
                        all_docs = set(index.documents.keys())
                        term_results = all_docs - term_results
                        negate_next = False
                    
                    # Apply the current operation
                    if current_op == 'AND':
                        if not results:  # If this is the first term, just take its results
                            results = term_results
                        else:
                            results &= term_results
                    else:  # OR
                        results |= term_results
                
                elif negate_next:
                    # If term doesn't exist but is negated, it's equivalent to all documents
                    all_docs = set(index.documents.keys())
                    results = all_docs if not results else results | all_docs
                    negate_next = False
                
                # Add step for visualization
                search_process['steps'].append({
                    'operation': f"Applied {current_op} with term '{token}'",
                    'term': token,
                    'term_docs': len(index.index.get(token, {})),
                    'result_count': len(results)
                })
        
        return results, pos
    
    # Evaluate the boolean expression
    results, _ = parse_expression(processed_tokens)
    
    # Convert to list and sort for consistent output
    result_list = sorted(list(results))
    
    # Add final results to search process
    search_process['results'] = result_list
    
    return result_list, search_process

def recommend_related_queries(query, index, max_suggestions=5):
    """
    Recommend related queries based on the original query
    
    Parameters:
    -----------
    query : str
        Original query
    index : InvertedIndex
        Inverted index to use for recommendations
    max_suggestions : int, default=5
        Maximum number of suggestions to return
        
    Returns:
    --------
    list
        List of suggested queries
    """
    # Process the query
    query_result = preprocess_text(query, do_stemming=index.use_stemming)
    query_tokens = query_result['stemmed'] if index.use_stemming else query_result['tokens']
    
    # Count co-occurring terms
    co_occurring_terms = {}
    
    for token in query_tokens:
        if token in index.index:
            # Get documents containing this term
            docs = index.index[token].keys()
            
            # Find terms that co-occur in these documents
            for doc_id in docs:
                # Count terms in this document
                doc_terms = {}
                for term, postings in index.index.items():
                    if doc_id in postings and term not in query_tokens:
                        doc_terms[term] = len(postings[doc_id]) if index.include_positions else postings[doc_id]
                
                # Add to co-occurring terms count
                for term, count in doc_terms.items():
                    if term in co_occurring_terms:
                        co_occurring_terms[term] += count
                    else:
                        co_occurring_terms[term] = count
    
    # Sort by frequency
    sorted_terms = sorted(co_occurring_terms.items(), key=lambda x: x[1], reverse=True)
    
    # Generate suggested queries
    suggestions = []
    for term, _ in sorted_terms[:max_suggestions]:
        # Add the term to the original query
        suggestion = f"{query} {term}"
        suggestions.append(suggestion)
    
    return suggestions

def visualize_search_results(results, documents, search_process=None, width=800, height=400):
    """
    Create a visualization of search results
    
    Parameters:
    -----------
    results : list
        List of (doc_id, score) tuples
    documents : dict
        Dictionary of document texts
    search_process : dict, optional
        Search process visualization data
    width : int, default=800
        Width of the visualization
    height : int, default=400
        Height of the visualization
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization figure
    """
    # Create a dataframe for visualization
    df = pd.DataFrame(results, columns=['Document', 'Score'])
    
    # Add document lengths
    df['Document Length'] = df['Document'].apply(lambda x: len(documents[x]))
    
    # Create a bar chart with Plotly
    fig = px.bar(
        df, 
        x='Document', 
        y='Score',
        color='Score',
        hover_data=['Document Length'],
        title='Search Results Scores',
        width=width,
        height=height,
        color_continuous_scale='Blues'
    )
    
    return fig

def streamlit_search_demo():
    """
    Streamlit demonstration of search functionality
    """
    st.title("Search Engines: Finding Needles in Digital Haystacks")
    
    st.markdown("""
    Search engines help us find relevant information from vast collections of documents.
    In this demonstration, we'll explore different search methods and see how bias can affect results.
    """)
    
    # Sample documents with diverse content
    sample_docs = {
        "classic_lit": """
        It is a truth universally acknowledged, that a single man in possession of a good fortune, 
        must be in want of a wife. However little known the feelings or views of such a man may be 
        on his first entering a neighbourhood, this truth is so well fixed in the minds of the 
        surrounding families, that he is considered the rightful property of some one or other of 
        their daughters.
        """,
        
        "modern_tech": """
        Artificial intelligence and machine learning have transformed how we interact with technology.
        Neural networks can now recognize images, process natural language, and even generate creative
        content like art and music. These advancements raise important ethical questions about the
        future of work and creativity.
        """,
        
        "cultural_history": """
        The Harlem Renaissance was an intellectual and cultural revival of African American music,
        dance, art, fashion, literature, and politics. It fostered new styles of expression and
        influenced American culture broadly. Artists like Langston Hughes, Zora Neale Hurston, and
        Louis Armstrong defined this era.
        """,
        
        "science_news": """
        Scientists have discovered a new species of deep-sea creatures living near hydrothermal vents.
        These organisms thrive in extreme conditions without sunlight, using chemosynthesis instead
        of photosynthesis. This finding expands our understanding of how life can adapt to harsh
        environments.
        """,
        
        "cultural_terms": """
        Many AAVE (African American Vernacular English) terms have entered mainstream vocabulary.
        Similarly, words from Latinx communities and cultures around the world have enriched English.
        Code-switching between languages and dialects is common in multilingual communities.
        """
    }
    
    # Create an inverted index
    st.subheader("Create Your Search Engine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preserve_case = st.checkbox("Preserve case", value=False)
        use_stemming = st.checkbox("Use stemming", value=True)
    
    with col2:
        include_positions = st.checkbox("Store word positions", value=True)
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
    
    # Initialize the index
    index = InvertedIndex(preserve_case, use_stemming, include_positions)
    
    # Add documents to the index
    for doc_id, text in sample_docs.items():
        index.add_document(doc_id, text)
    
    # Search interface
    st.subheader("Search Documents")
    
    search_type = st.radio("Search method:", ["Standard Search", "Phrase Search", "Boolean Search"])
    
    # Provide search examples based on type
    if search_type == "Standard Search":
        examples = ["artificial intelligence", "cultural", "renaissance language", "science"]
    elif search_type == "Phrase Search":
        examples = ["artificial intelligence", "African American", "deep-sea creatures", "universally acknowledged"]
    else:  # Boolean Search
        examples = ["cultural AND language", "renaissance OR revival", "science AND NOT artificial"]
    
    # Show examples
    st.write("Example queries:")
    example_cols = st.columns(len(examples))
    
    for i, example in enumerate(examples):
        with example_cols[i]:
            if st.button(example):
                query = example
    
    # Get the query
    query = st.text_input("Enter your search query:", value=examples[0] if 'query' not in locals() else query)
    
    if query:
        # Perform search based on type
        if search_type == "Standard Search":
            results, process = index.search(query)
        elif search_type == "Phrase Search":
            results, process = index.phrase_search(query)
        else:  # Boolean Search
            results, process = boolean_search(query, index)
            # Convert to the format expected by the visualization
            results = [(doc_id, 1.0) for doc_id in results]
        
        # Display results
        st.write(f"### Search Results for: '{query}'")
        
        if results:
            for i, (doc_id, score) in enumerate(results, 1):
                st.write(f"**{i}. {doc_id}** (Score: {score:.4f})")
                st.write(f"> {index.get_document_snippet(doc_id)}")
            
            # Visualize results
            fig = visualize_search_results(results, index.documents, process)
            st.plotly_chart(fig)
            
            # Recommend related queries
            if search_type == "Standard Search":
                st.subheader("You might also want to search for:")
                suggestions = recommend_related_queries(query, index)
                
                for suggestion in suggestions:
                    if st.button(suggestion):
                        query = suggestion
        else:
            st.write("No results found for your query.")
        
        # Show search process details
        with st.expander("Show Search Process Details"):
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
                    st.write(f"- Found in documents: {', '.join(str(d) for d in data['docs'])}")
            
            elif search_type == "Phrase Search":
                # Phrase search process visualization
                st.write("#### Phrase Processing")
                st.write(f"Original phrase: '{process['phrase']}'")
                st.write(f"Processed phrase tokens: {process['processed_phrase']}")
                
                # Show matching details
                if process['matching_details']:
                    st.write("#### Matching Details")
                    for doc_id, details in process['matching_details'].items():
                        st.write(f"**Document '{doc_id}'**")
                        st.write(f"- Matches found at positions: {details['match_positions']}")
                        
                        # Show context around match
                        context = details['context']
                        st.write("- Context around first match:")
                        st.write(f"  ...{context['before']} **{context['match']}** {context['after']}...")
            
            else:  # Boolean Search
                # Boolean search process visualization
                st.write("#### Boolean Query Processing")
                st.write(f"Original query: '{process['query']}'")
                st.write(f"Tokens: {process['tokens']}")
                st.write(f"Processed tokens: {process['processed_tokens']}")
                
                # Show term mapping
                st.write("#### Term Mapping")
                for original, processed in process['term_mapping'].items():
                    st.write(f"'{original}' → '{processed}'")
                
                # Show evaluation steps
                st.write("#### Evaluation Steps")
                for i, step in enumerate(process['steps'], 1):
                    st.write(f"**Step {i}:** {step['operation']}")
                    if 'term' in step:
                        st.write(f"- Term '{step['term']}' found in {step['term_docs']} documents")
                    st.write(f"- Result set now contains {step['result_count']} documents")
    
    # Bias analysis
    st.subheader("Bias Analysis")
    
    st.markdown("""
    Search engines can reinforce existing biases in several ways:
    
    1. **Corpus bias**: If certain topics or perspectives are over/under-represented in the document collection
    2. **Query understanding bias**: How systems interpret and process queries
    3. **Ranking bias**: How results are scored and ordered
    
    Let's explore these biases with some experiments.
    """)
    
    # Experiment 1: Cultural term search
    st.write("#### Experiment 1: Cultural Term Representation")
    
    cultural_terms = ["Harlem", "Renaissance", "AAVE", "Latinx", "African American"]
    general_terms = ["science", "technology", "literature", "history", "art"]
    
    # Compare cultural vs general terms
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cultural Terms Results**")
        cultural_results = []
        
        for term in cultural_terms:
            results, _ = index.search(term)
            doc_count = len(results)
            cultural_results.append((term, doc_count))
        
        cultural_df = pd.DataFrame(cultural_results, columns=['Term', 'Documents Found'])
        st.dataframe(cultural_df)
    
    with col2:
        st.write("**General Terms Results**")
        general_results = []
        
        for term in general_terms:
            results, _ = index.search(term)
            doc_count = len(results)
            general_results.append((term, doc_count))
        
        general_df = pd.DataFrame(general_results, columns=['Term', 'Documents Found'])
        st.dataframe(general_df)
    
    # Experiment 2: Stemming effects
    st.write("#### Experiment 2: Stemming Effects on Cultural Terms")
    
    # Terms that might be affected differently by stemming
    term_pairs = [
        ("African", "Africans"),
        ("American", "Americans"),
        ("language", "languages"),
        ("Latinx", "Latino"),
        ("community", "communities")
    ]
    
    # Compare stemmed vs. unstemmed search
    stem_results = []
    
    for term1, term2 in term_pairs:
        # With stemming
        index_with_stem = InvertedIndex(preserve_case=False, use_stemming=True)
        for doc_id, text in sample_docs.items():
            index_with_stem.add_document(doc_id, text)
        
        results1_stem, _ = index_with_stem.search(term1)
        results2_stem, _ = index_with_stem.search(term2)
        
        # Without stemming
        index_no_stem = InvertedIndex(preserve_case=False, use_stemming=False)
        for doc_id, text in sample_docs.items():
            index_no_stem.add_document(doc_id, text)
        
        results1_no_stem, _ = index_no_stem.search(term1)
        results2_no_stem, _ = index_no_stem.search(term2)
        
        # Add to results
        stem_results.append({
            'Term 1': term1,
            'Term 2': term2,
            'With Stemming - Same Results': set([r[0] for r in results1_stem]) == set([r[0] for r in results2_stem]),
            'Without Stemming - Same Results': set([r[0] for r in results1_no_stem]) == set([r[0] for r in results2_no_stem])
        })
    
    stem_df = pd.DataFrame(stem_results)
    st.dataframe(stem_df)
    
    # Reflection questions
    st.subheader("Reflection Questions")
    
    st.markdown("""
    1. How might the representation of different cultural terms in the corpus affect search results?
    2. What information is lost during preprocessing (case folding, stemming) that might be important for cultural context?
    3. How might TF-IDF scoring disadvantage certain terms or topics?
    4. How would you design a more equitable search system that reduces these biases?
    """)

if __name__ == "__main__":
    # This allows the file to be run directly as a Streamlit app
    streamlit_search_demo()