# utils/tfidf.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import re
import math
from collections import Counter

def compute_tf(text, term):
    """
    Compute Term Frequency for educational visualization
    
    Parameters:
    -----------
    text : str
        Document text
    term : str
        Term to compute TF for
        
    Returns:
    --------
    float
        Term frequency
    dict
        Breakdown of calculation for educational purposes
    """
    # Preprocess text for matching
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    
    # Count occurrences
    term = term.lower()
    term_count = words.count(term)
    total_words = len(words)
    
    # Calculate TF
    tf = term_count / total_words if total_words > 0 else 0
    
    # Breakdown for educational purposes
    breakdown = {
        'term_count': term_count,
        'total_words': total_words,
        'calculation': f"TF = {term_count} / {total_words} = {tf:.4f}",
        'explanation': "Term Frequency (TF) measures how frequently a term occurs in a document."
    }
    
    return tf, breakdown

def compute_idf(corpus, term):
    """
    Compute Inverse Document Frequency for educational visualization
    
    Parameters:
    -----------
    corpus : list
        List of document texts
    term : str
        Term to compute IDF for
        
    Returns:
    --------
    float
        Inverse document frequency
    dict
        Breakdown of calculation for educational purposes
    """
    # Count documents containing the term
    term = term.lower()
    doc_count = sum(1 for doc in corpus if term in doc.lower())
    total_docs = len(corpus)
    
    # Calculate IDF
    idf = math.log((total_docs + 1) / (doc_count + 1)) + 1  # Smoothed IDF
    
    # Breakdown for educational purposes
    breakdown = {
        'doc_count': doc_count,
        'total_docs': total_docs,
        'calculation': f"IDF = log(({total_docs} + 1)/({doc_count} + 1)) + 1 = {idf:.4f}",
        'explanation': "Inverse Document Frequency (IDF) measures how important a term is across the entire corpus."
    }
    
    return idf, breakdown

def compute_tfidf(corpus, doc_idx, term):
    """
    Compute TF-IDF with explanation for educational visualization
    
    Parameters:
    -----------
    corpus : list
        List of document texts
    doc_idx : int
        Index of the target document in the corpus
    term : str
        Term to compute TF-IDF for
        
    Returns:
    --------
    dict
        Complete TF-IDF calculation with explanations
    """
    # Compute TF
    tf, tf_breakdown = compute_tf(corpus[doc_idx], term)
    
    # Compute IDF
    idf, idf_breakdown = compute_idf(corpus, term)
    
    # Calculate TF-IDF
    tfidf = tf * idf
    
    # Complete breakdown
    result = {
        'term': term,
        'document': doc_idx,
        'tf': tf,
        'idf': idf,
        'tfidf': tfidf,
        'tf_breakdown': tf_breakdown,
        'idf_breakdown': idf_breakdown,
        'calculation': f"TF-IDF = {tf:.4f} × {idf:.4f} = {tfidf:.4f}",
        'explanation': "TF-IDF balances how frequent a term is in a document with how rare it is across all documents."
    }
    
    return result

def compare_terms_tfidf(corpus, terms, doc_indices=None):
    """
    Compare TF-IDF scores for multiple terms across documents
    
    Parameters:
    -----------
    corpus : list
        List of document texts
    terms : list
        List of terms to compare
    doc_indices : list, optional
        List of document indices to include in comparison
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with TF-IDF scores for visualization
    """
    if doc_indices is None:
        doc_indices = list(range(len(corpus)))
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Prepare data for plotting
    results = []
    
    for term in terms:
        term_lower = term.lower()
        
        # Find the term in feature names
        if term_lower in feature_names:
            term_idx = list(feature_names).index(term_lower)
            
            for doc_idx in doc_indices:
                # Get TF-IDF value
                tfidf_value = tfidf_matrix[doc_idx, term_idx]
                
                # Add to results
                results.append({
                    'term': term,
                    'document': f"Document {doc_idx + 1}",
                    'tfidf': tfidf_value
                })
        else:
            # Term not in vocabulary
            for doc_idx in doc_indices:
                results.append({
                    'term': term,
                    'document': f"Document {doc_idx + 1}",
                    'tfidf': 0.0
                })
    
    return pd.DataFrame(results)

def visualize_tfidf_comparison(df):
    """
    Create a visualization of TF-IDF comparison
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with TF-IDF scores
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure object for the visualization
    """
    # Create heatmap
    fig = px.density_heatmap(
        df, 
        x='document', 
        y='term', 
        z='tfidf',
        color_continuous_scale='Blues',
        title='TF-IDF Comparison Across Terms and Documents'
    )
    
    # Add text annotations
    annotations = []
    for i, row in df.iterrows():
        annotations.append(dict(
            x=row['document'],
            y=row['term'],
            text=f"{row['tfidf']:.4f}",
            showarrow=False,
            font=dict(color='black' if row['tfidf'] < 0.5 else 'white')
        ))
    
    fig.update_layout(annotations=annotations)
    
    return fig

def visualize_tfidf_breakdown(result):
    """
    Create a visual breakdown of TF-IDF calculation
    
    Parameters:
    -----------
    result : dict
        Result from compute_tfidf
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Figure object for the visualization
    """
    # Create a figure with subplots
    fig = go.Figure()
    
    # Add TF bar
    fig.add_trace(go.Bar(
        x=['Term Frequency (TF)'],
        y=[result['tf']],
        name='TF',
        marker_color='lightblue',
        text=[f"{result['tf']:.4f}"],
        textposition='auto'
    ))
    
    # Add IDF bar
    fig.add_trace(go.Bar(
        x=['Inverse Document Frequency (IDF)'],
        y=[result['idf']],
        name='IDF',
        marker_color='lightgreen',
        text=[f"{result['idf']:.4f}"],
        textposition='auto'
    ))
    
    # Add TF-IDF bar
    fig.add_trace(go.Bar(
        x=['TF-IDF Score'],
        y=[result['tfidf']],
        name='TF-IDF',
        marker_color='coral',
        text=[f"{result['tfidf']:.4f}"],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"TF-IDF Calculation Breakdown for '{result['term']}'",
        xaxis=dict(title='Component'),
        yaxis=dict(title='Value'),
        barmode='group'
    )
    
    return fig

def cultural_vs_general_terms_analysis():
    """
    Analyze differences between cultural and general terms in TF-IDF scores
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with analysis results
    """
    # Sample texts representing different cultural contexts
    corpus = [
        # Western-centric text
        "The democracy and freedom are fundamental values in our society. The industrial revolution changed how we view economics and capitalism.",
        
        # Text with African-American cultural references
        "The community gathered for Kwanzaa celebrations. Many discussed the importance of afrofuturism in modern literature and hip-hop culture.",
        
        # Text with Latin American cultural references
        "The quinceañera celebration was beautiful. Many latinx writers explore themes of mestizaje and borderlands in their work.",
        
        # Text with Asian cultural references
        "The lunar new year festival included traditional performances. Many discussed the concept of filial piety in contemporary society."
    ]
    
    # Define term sets
    general_terms = ['society', 'important', 'modern', 'traditional', 'celebration']
    cultural_terms = ['afrofuturism', 'hip-hop', 'latinx', 'mestizaje', 'quinceañera', 'kwanzaa', 'filial']
    
    # Compare terms
    results = compare_terms_tfidf(corpus, general_terms + cultural_terms)
    
    # Add term type
    results['term_type'] = results['term'].apply(
        lambda x: 'Cultural Term' if x in cultural_terms else 'General Term'
    )
    
    return results

def streamlit_tfidf_module():
    """
    Streamlit module for TF-IDF demonstration
    """
    st.title("Understanding TF-IDF: Numbers That Shape Results")
    
    st.markdown("""
    Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used to reflect 
    how important a word is to a document in a collection. It's one of the fundamental algorithms 
    used in search engines to rank results.
    
    Let's explore how TF-IDF works and how it might introduce bias.
    """)
    
    # Sample documents
    sample_docs = {
        "Pride and Prejudice (excerpt)": """
        It is a truth universally acknowledged, that a single man in possession of a good fortune, 
        must be in want of a wife. However little known the feelings or views of such a man may be 
        on his first entering a neighbourhood, this truth is so well fixed in the minds of the 
        surrounding families, that he is considered the rightful property of some one or other of 
        their daughters.
        """,
        
        "Moby Dick (excerpt)": """
        Call me Ishmael. Some years ago—never mind how long precisely—having little or no money 
        in my purse, and nothing particular to interest me on shore, I thought I would sail about 
        a little and see the watery part of the world. It is a way I have of driving off the spleen 
        and regulating the circulation.
        """,
        
        "The Art of War (excerpt)": """
        The art of war is of vital importance to the State. It is a matter of life and death, 
        a road either to safety or to ruin. Hence it is a subject of inquiry which can on no account 
        be neglected. The art of war, then, is governed by five constant factors, to be taken into 
        account in one's deliberations, when seeking to determine the conditions obtaining in the field.
        """,
        
        "Their Eyes Were Watching God (excerpt)": """
        Ships at a distance have every man's wish on board. For some they come in with the tide. 
        For others they sail forever on the horizon, never out of sight, never landing until the 
        Watcher turns his eyes away in resignation, his dreams mocked to death by Time. That is 
        the life of men. Now, women forget all those things they don't want to remember, and 
        remember everything they don't want to forget. The dream is the truth.
        """
    }
    
    corpus = list(sample_docs.values())
    doc_names = list(sample_docs.keys())
    
    # TF-IDF Calculator
    st.subheader("TF-IDF Calculator")
    
    # Word input
    word = st.text_input("Enter a word:", "love")
    
    # Document selection
    selected_docs = st.multiselect(
        "Select documents to compare:",
        options=doc_names,
        default=[doc_names[0]]
    )
    
    if word and selected_docs:
        # Get document indices
        doc_indices = [doc_names.index(doc) for doc in selected_docs]
        
        # Calculate TF-IDF for each selected document
        results = []
        for doc_idx in doc_indices:
            result = compute_tfidf(corpus, doc_idx, word)
            results.append(result)
        
        # Display TF-IDF scores
        st.write(f"TF-IDF scores for '{word}':")
        
        # Create a table for comparison
        comparison_data = {
            'Document': selected_docs,
            'TF': [result['tf'] for result in results],
            'IDF': [result['idf'] for result in results],
            'TF-IDF': [result['tfidf'] for result in results]
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Create bar chart visualization
        fig = px.bar(
            comparison_df, 
            x='Document', 
            y='TF-IDF',
            title=f"TF-IDF Scores for '{word}'",
            color='TF-IDF',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig)
        
        # Show calculation breakdown for the first selected document
        st.subheader("Calculation Breakdown")
        st.markdown(f"### How is TF-IDF calculated for '{word}' in {selected_docs[0]}?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Term Frequency (TF)")
            st.info(results[0]['tf_breakdown']['explanation'])
            st.code(results[0]['tf_breakdown']['calculation'])
            
            st.markdown("#### Inverse Document Frequency (IDF)")
            st.info(results[0]['idf_breakdown']['explanation'])
            st.code(results[0]['idf_breakdown']['calculation'])
            
            st.markdown("#### TF-IDF Score")
            st.success(results[0]['calculation'])
        
        with col2:
            # Visual breakdown
            breakdown_fig = visualize_tfidf_breakdown(results[0])
            st.plotly_chart(breakdown_fig)
    
    # Cultural vs General Terms Analysis
    st.subheader("Cultural vs. General Terms Analysis")
    
    st.markdown("""
    TF-IDF calculations can treat culturally specific terms differently from general terms. 
    Let's explore how this might introduce bias in search results.
    """)
    
    # Sample analysis
    analysis_results = cultural_vs_general_terms_analysis()
    
    # Display average TF-IDF by term type
    avg_by_type = analysis_results.groupby('term_type')['tfidf'].mean().reset_index()
    
    fig = px.bar(
        avg_by_type,
        x='term_type',
        y='tfidf',
        title='Average TF-IDF Score by Term Type',
        color='term_type',
        labels={'tfidf': 'Average TF-IDF Score', 'term_type': 'Term Type'}
    )
    st.plotly_chart(fig)
    
    # Display heatmap of all terms
    heatmap_fig = visualize_tfidf_comparison(analysis_results)
    st.plotly_chart(heatmap_fig)
    
    # Hypothesis box
    st.subheader("Form a Hypothesis")
    st.markdown("""
    **How might TF-IDF scores differ for cultural terms vs. general terms?**
    
    Try entering different types of words in the calculator above:
    - Common English words (e.g., "love", "time", "day")
    - Cultural terms (e.g., "diaspora", "intersectionality")
    - Technical terms from different fields
    
    What patterns do you notice? What biases might this introduce in search results?
    """)
    
    hypothesis = st.text_area("Your hypothesis about TF-IDF bias:", height=150)
    
    # Provide some insights
    if st.button("Show Insights"):
        st.markdown("""
        ### Key Insights about TF-IDF and Bias
        
        1. **Rare terms get higher IDF scores** - Cultural terms that appear rarely in a corpus get higher IDF scores, 
        which might seem like an advantage. However, this can be misleading because:
        
        2. **Corpus representation matters** - If a corpus lacks diverse cultural representation, certain terms may 
        appear artificially rare, not because they're unimportant but because the corpus itself is biased.
        
        3. **Stemming effects** - Preprocessing steps like stemming may handle terms from different cultures inconsistently, 
        affecting their TF-IDF scores.
        
        4. **Term frequency thresholds** - Many search engines ignore terms below certain frequency thresholds, 
        which can disproportionately affect cultural terms in unbalanced corpora.
        
        5. **Context loss** - TF-IDF treats terms as independent units, losing important cultural context and associations.
        """)

if __name__ == "__main__":
    streamlit_tfidf_module()