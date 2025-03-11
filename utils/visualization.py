# utils/visualization.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import re
import nltk
from nltk.stem import PorterStemmer
import json
import os

def plot_corpus_stats(docs, metadata=None):
    """
    Create visualizations of corpus statistics
    
    Parameters:
    -----------
    docs : dict
        Dictionary of document texts
    metadata : dict, optional
        Dictionary of document metadata
        
    Returns:
    --------
    dict
        Dictionary of Plotly figures
    """
    figures = {}
    
    # Document lengths
    doc_lengths = {doc_id: len(text) for doc_id, text in docs.items()}
    
    # Sort by length for better visualization
    sorted_lengths = sorted(doc_lengths.items(), key=lambda x: x[1], reverse=True)
    
    # Create bar chart for document lengths
    df_lengths = pd.DataFrame(sorted_lengths, columns=['Document', 'Length'])
    fig_lengths = px.bar(
        df_lengths, 
        x='Document', 
        y='Length',
        title='Document Lengths',
        color='Length',
        color_continuous_scale='Blues'
    )
    figures['doc_lengths'] = fig_lengths
    
    # Unique words per document
    doc_unique_words = {}
    
    for doc_id, text in docs.items():
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        unique_words = len(set(tokens))
        doc_unique_words[doc_id] = unique_words
    
    # Sort by unique word count
    sorted_unique = sorted(doc_unique_words.items(), key=lambda x: x[1], reverse=True)
    
    # Create bar chart for unique words
    df_unique = pd.DataFrame(sorted_unique, columns=['Document', 'Unique Words'])
    fig_unique = px.bar(
        df_unique, 
        x='Document', 
        y='Unique Words',
        title='Unique Words per Document',
        color='Unique Words',
        color_continuous_scale='Greens'
    )
    figures['unique_words'] = fig_unique
    
    # If metadata is provided, create additional visualizations
    if metadata:
        # Try to extract years or dates
        years = {}
        for doc_id, meta in metadata.items():
            # Look for year in metadata
            year = None
            for key, value in meta.items():
                if key.lower() in ('year', 'date', 'publication_year', 'pub_year'):
                    try:
                        # Extract year from date if it's a full date
                        if isinstance(value, str):
                            year_match = re.search(r'\b(18|19|20)\d{2}\b', value)
                            if year_match:
                                year = int(year_match.group())
                        elif isinstance(value, (int, float)):
                            year = int(value)
                    except:
                        pass
            
            if year:
                years[doc_id] = year
        
        # Create histogram of years if available
        if years:
            years_list = list(years.values())
            fig_years = px.histogram(
                x=years_list,
                nbins=20,
                title='Documents by Year',
                labels={'x': 'Year', 'y': 'Count'},
                color_discrete_sequence=['teal']
            )
            figures['years'] = fig_years
        
        # Try to extract author demographics
        author_genders = {}
        author_nationalities = {}
        
        for doc_id, meta in metadata.items():
            # Look for gender in metadata
            for key, value in meta.items():
                if key.lower() in ('gender', 'author_gender', 'sex'):
                    if isinstance(value, str):
                        author_genders[doc_id] = value.strip().lower()
                
                if key.lower() in ('nationality', 'country', 'origin', 'author_nationality'):
                    if isinstance(value, str):
                        author_nationalities[doc_id] = value.strip()
        
        # Create pie charts for demographics if available
        if author_genders:
            gender_counts = Counter(author_genders.values())
            fig_genders = px.pie(
                names=list(gender_counts.keys()),
                values=list(gender_counts.values()),
                title='Author Gender Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            figures['gender'] = fig_genders
        
        if author_nationalities:
            nationality_counts = Counter(author_nationalities.values())
            # Sort by count for better visualization
            sorted_nationalities = sorted(nationality_counts.items(), key=lambda x: x[1], reverse=True)
            nationality_df = pd.DataFrame(sorted_nationalities, columns=['Nationality', 'Count'])
            
            fig_nationalities = px.bar(
                nationality_df,
                x='Nationality',
                y='Count',
                title='Author Nationality Distribution',
                color='Count',
                color_continuous_scale='Viridis'
            )
            figures['nationality'] = fig_nationalities
    
    return figures

def visualize_preprocessing_steps(text, do_stemming=True, keep_case=False, remove_stopwords=True):
    """
    Create visualizations of text preprocessing steps
    
    Parameters:
    -----------
    text : str
        Original text
    do_stemming : bool, default=True
        Whether to apply stemming
    keep_case : bool, default=False
        Whether to preserve case information
    remove_stopwords : bool, default=True
        Whether to remove stopwords
        
    Returns:
    --------
    dict
        Dictionary of visualization data
    """
    # Ensure nltk resources are available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    # Initialize visualization data
    viz_data = {
        'original': text,
        'steps': []
    }
    
    # Step 1: Lowercase (optional)
    if not keep_case:
        lowercase_text = text.lower()
        viz_data['steps'].append({
            'name': 'Lowercase Conversion',
            'description': 'Convert all text to lowercase to standardize words regardless of capitalization.',
            'input': text,
            'output': lowercase_text,
            'highlighted_diffs': _highlight_case_differences(text, lowercase_text),
            'bias_implications': 'Case folding loses distinction between proper nouns and common words. This affects named entities, cultural terms, and acronyms.'
        })
        current_text = lowercase_text
    else:
        current_text = text
    
    # Step 2: Punctuation removal
    no_punct_text = re.sub(r'[^\w\s]', '', current_text)
    viz_data['steps'].append({
        'name': 'Punctuation Removal',
        'description': 'Remove punctuation marks that might interfere with token matching.',
        'input': current_text,
        'output': no_punct_text,
        'highlighted_diffs': _highlight_punctuation(current_text),
        'bias_implications': 'Removing punctuation affects apostrophes in names (O\'Connor), hyphens in compound terms (African-American), and special characters in loanwords (café).'
    })
    current_text = no_punct_text
    
    # Step 3: Tokenization
    tokens = nltk.word_tokenize(current_text)
    viz_data['steps'].append({
        'name': 'Tokenization',
        'description': 'Split text into individual words (tokens).',
        'input': current_text,
        'output': str(tokens),
        'bias_implications': 'Tokenization may split compounds incorrectly, particularly affecting non-English terms.'
    })
    current_tokens = tokens
    
    # Step 4: Stopword removal (optional)
    if remove_stopwords:
        stopwords_set = set(nltk.corpus.stopwords.words('english'))
        filtered_tokens = [token for token in current_tokens if token.lower() not in stopwords_set]
        viz_data['steps'].append({
            'name': 'Stopword Removal',
            'description': 'Remove common words (like "the", "and", "is") that typically don\'t carry much meaning for search.',
            'input': str(current_tokens),
            'output': str(filtered_tokens),
            'highlighted_diffs': _highlight_stopwords(current_tokens),
            'bias_implications': 'Stopwords sometimes carry meaning in certain phrases or cultural expressions. English stopword lists may inappropriately apply to text containing multiple languages.'
        })
        current_tokens = filtered_tokens
    
    # Step 5: Stemming (optional)
    if do_stemming:
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in current_tokens]
        viz_data['steps'].append({
            'name': 'Stemming',
            'description': 'Reduce words to their root form to match similar words (e.g., "running" → "run").',
            'input': str(current_tokens),
            'output': str(stemmed_tokens),
            'highlighted_diffs': _highlight_stemming_differences(current_tokens, stemmed_tokens),
            'bias_implications': 'Stemming algorithms are typically designed for English and can incorrectly stem words from other languages or cultural terms. Different inflections may have distinct meanings in some contexts.'
        })
        current_tokens = stemmed_tokens
    
    # Final processed text
    processed_text = ' '.join(current_tokens)
    viz_data['processed'] = processed_text
    
    # Overall information loss assessment
    original_tokens = nltk.word_tokenize(text.lower())
    information_loss = {
        'original_token_count': len(original_tokens),
        'final_token_count': len(current_tokens),
        'token_reduction_percentage': (1 - len(current_tokens) / len(original_tokens)) * 100 if len(original_tokens) > 0 else 0,
        'unique_tokens_original': len(set(original_tokens)),
        'unique_tokens_final': len(set(current_tokens))
    }
    viz_data['information_loss'] = information_loss
    
    return viz_data

def _highlight_case_differences(original, lowercase):
    """Helper function to highlight case differences"""
    highlights = []
    for i, char in enumerate(original):
        if char.lower() == char:
            continue
        
        # Find the word containing this capital letter
        word_match = re.search(r'\b\w*' + re.escape(char) + r'\w*\b', original[max(0, i-10):min(len(original), i+10)])
        if word_match:
            word = word_match.group()
            highlights.append({
                'original': word,
                'processed': word.lower(),
                'type': 'case_folding'
            })
    
    # Limit to 10 examples
    return highlights[:10]

def _highlight_punctuation(text):
    """Helper function to highlight punctuation"""
    punct_matches = re.finditer(r'[^\w\s]', text)
    highlights = []
    
    for match in punct_matches:
        # Get some context around the punctuation
        start = max(0, match.start() - 10)
        end = min(len(text), match.end() + 10)
        context = text[start:end]
        
        # Find the word containing or adjacent to this punctuation
        context_without_punct = re.sub(r'[^\w\s]', '', context)
        
        highlights.append({
            'original': context,
            'processed': context_without_punct,
            'type': 'punctuation_removal'
        })
        
        # Limit to 10 examples
        if len(highlights) >= 10:
            break
    
    return highlights

def _highlight_stopwords(tokens):
    """Helper function to highlight stopwords"""
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    highlights = []
    
    for token in tokens:
        if token.lower() in stopwords_set:
            highlights.append({
                'original': token,
                'processed': '[REMOVED]',
                'type': 'stopword_removal'
            })
    
    # Limit to 10 examples
    return highlights[:10]

def _highlight_stemming_differences(original_tokens, stemmed_tokens):
    """Helper function to highlight stemming differences"""
    highlights = []
    
    for orig, stemmed in zip(original_tokens, stemmed_tokens):
        if orig != stemmed:
            highlights.append({
                'original': orig,
                'processed': stemmed,
                'type': 'stemming'
            })
    
    # Limit to 10 examples
    return highlights[:10]

def plot_term_frequency_distribution(docs, top_n=20):
    """
    Create visualization of term frequency distribution across documents
    
    Parameters:
    -----------
    docs : dict
        Dictionary of document texts
    top_n : int, default=20
        Number of top terms to show
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Term frequency heatmap
    """
    # Tokenize documents
    doc_tokens = {}
    for doc_id, text in docs.items():
        tokens = re.findall(r'\b\w+\b', text.lower())
        doc_tokens[doc_id] = tokens
    
    # Count terms across all documents
    all_terms = []
    for tokens in doc_tokens.values():
        all_terms.extend(tokens)
    
    term_counts = Counter(all_terms)
    
    # Get top N terms
    top_terms = [term for term, count in term_counts.most_common(top_n)]
    
    # Create frequency matrix
    frequency_data = []
    
    for term in top_terms:
        for doc_id, tokens in doc_tokens.items():
            term_freq = tokens.count(term)
            
            # Add to data
            frequency_data.append({
                'Term': term,
                'Document': doc_id,
                'Frequency': term_freq
            })
    
    # Create dataframe
    df = pd.DataFrame(frequency_data)
    
    # Create heatmap
    fig = px.density_heatmap(
        df,
        x='Document',
        y='Term',
        z='Frequency',
        title=f'Top {top_n} Term Frequencies Across Documents',
        color_continuous_scale='Blues'
    )
    
    return fig

def visualize_cultural_term_comparison(texts, cultural_terms, general_terms):
    """
    Compare how cultural terms vs general terms are represented in texts
    
    Parameters:
    -----------
    texts : dict
        Dictionary of text documents
    cultural_terms : list
        List of cultural terms to analyze
    general_terms : list
        List of general terms to analyze
        
    Returns:
    --------
    dict
        Dictionary of visualization figures
    """
    figures = {}
    
    # Count term occurrences
    term_counts = {}
    
    for term in cultural_terms + general_terms:
        counts = {}
        for doc_id, text in texts.items():
            # Simple case-insensitive count
            count = len(re.findall(r'\b' + re.escape(term.lower()) + r'\b', text.lower()))
            counts[doc_id] = count
        
        term_counts[term] = counts
    
    # Create dataframe for comparison
    comparison_data = []
    
    for term, counts in term_counts.items():
        term_type = 'Cultural Term' if term in cultural_terms else 'General Term'
        doc_count = sum(1 for count in counts.values() if count > 0)
        total_count = sum(counts.values())
        
        comparison_data.append({
            'Term': term,
            'Type': term_type,
            'Documents Found': doc_count,
            'Total Occurrences': total_count,
            'Avg. per Document': total_count / len(texts) if texts else 0
        })
    
    # Create dataframe
    df = pd.DataFrame(comparison_data)
    
    # Create comparison bar chart
    fig_docs = px.bar(
        df,
        x='Term',
        y='Documents Found',
        color='Type',
        title='Number of Documents Containing Each Term',
        barmode='group',
        color_discrete_map={'Cultural Term': 'coral', 'General Term': 'skyblue'}
    )
    figures['doc_count'] = fig_docs
    
    # Create occurrences bar chart
    fig_occurrences = px.bar(
        df,
        x='Term',
        y='Total Occurrences',
        color='Type',
        title='Total Occurrences of Each Term',
        barmode='group',
        color_discrete_map={'Cultural Term': 'coral', 'General Term': 'skyblue'}
    )
    figures['occurrences'] = fig_occurrences
    
    # Create averages box plot
    fig_avg = px.box(
        df,
        x='Type',
        y='Avg. per Document',
        title='Average Occurrences per Document by Term Type',
        color='Type',
        points='all',
        color_discrete_map={'Cultural Term': 'coral', 'General Term': 'skyblue'}
    )
    figures['averages'] = fig_avg
    
    return figures

def visualize_stemming_effects(terms, use_porter=True):
    """
    Visualize how stemming affects different terms
    
    Parameters:
    -----------
    terms : list
        List of terms to analyze
    use_porter : bool, default=True
        Whether to use Porter stemmer (or alternative)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Visualization of stemming effects
    """
    # Initialize stemmers
    porter = PorterStemmer()
    
    # Get stems for each term
    stemming_results = []
    
    for term in terms:
        porter_stem = porter.stem(term)
        
        # Check if stemming changed the term
        porter_changed = porter_stem != term
        
        stemming_results.append({
            'Term': term,
            'Porter Stem': porter_stem,
            'Changed by Porter': porter_changed
        })
    
    # Create dataframe
    df = pd.DataFrame(stemming_results)
    
    # Create sankey diagram for term -> stem flow
    labels = list(df['Term']) + list(df['Porter Stem'].unique())
    
    # Create source-target pairs for the flow
    sources = []
    targets = []
    values = []
    
    # Get unique target labels (stems)
    unique_stems = list(df['Porter Stem'].unique())
    
    # For each term, add a flow to its stem
    for i, row in df.iterrows():
        term_idx = i
        stem_idx = len(df) + unique_stems.index(row['Porter Stem'])
        
        sources.append(term_idx)
        targets.append(stem_idx)
        values.append(1)  # Each term has equal weight
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["rgba(255, 99, 71, 0.8)"] * len(df) + ["rgba(135, 206, 235, 0.8)"] * len(unique_stems)
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])
    
    fig.update_layout(title_text="How Stemming Maps Terms to Common Stems")
    
    return fig

def visualize_bias_entry_points(highlights=True):
    """
    Create a visualization of the IR flowchart with bias entry points
    
    Parameters:
    -----------
    highlights : bool, default=True
        Whether to highlight bias entry points
        
    Returns:
    --------
    str
        Mermaid flowchart code
    """
    flowchart = """
    flowchart TD
        subgraph Input[Input and Preprocessing]
            A[Document Collection] -->|Selection Bias| B[Raw Text]
            style A fill:#ffcccc
            B --> C[Text Preprocessing]
            C -->|Stemming Bias| D[Processed Text]
            style C fill:#ffcccc
        end
        subgraph Indexing[Indexing System]
            D --> E[Build Inverted Index]
            E -->|Context Loss| F[Word-Document Mapping]
            style E fill:#ffcccc
            F --> F1[Positional Index]
        end
        subgraph Query[Query Processing]
            G[User Query] --> H[Query Preprocessing]
            H -->|Same Stemming Bias| I[Processed Query]
            style H fill:#ffcccc
        end
        subgraph Retrieval[Retrieval Methods]
            I --> J1[Boolean Retrieval]
            F --> J1
            J1 -->|Exact Match Bias| K1[Boolean Results]
            style J1 fill:#ffcccc
            I --> J2[Phrase Query Retrieval]
            F1 --> J2
            J2 -->|Strict Order Bias| K2[Phrase Results]
            style J2 fill:#ffcccc
            I --> J3[TF-IDF Calculation]
            F --> J3
            J3 -->|Statistical Bias| K3[TF-IDF Vectors]
            style J3 fill:#ffcccc
            K3 --> L[Cosine Similarity]
            style L fill:#ffcccc
            L -->|Vector Space Bias| K4[Similarity Results]
        end
        subgraph Ranking[Result Integration]
            K1 --> M[Result Integration]
            K2 --> M
            K4 --> M
            M --> N[Final Search Results]
        end
        classDef biasNode fill:#ffcccc,stroke:#ff0000
        classDef normalNode fill:#white
        linkStyle default stroke:#333,stroke-width:2px
    """
    
    if not highlights:
        # Remove highlighting styles
        flowchart = re.sub(r'style \w+ fill:#ffcccc', '', flowchart)
        flowchart = re.sub(r'classDef biasNode fill:#ffcccc,stroke:#ff0000', '', flowchart)
    
    return flowchart

def visualize_search_process(search_process):
    """
    Create visualizations of a search process
    
    Parameters:
    -----------
    search_process : dict
        Search process data from search function
        
    Returns:
    --------
    dict
        Dictionary of visualization figures
    """
    figures = {}
    
    # If this is a standard search
    if 'token_postings' in search_process:
        # Create token frequency visualization
        token_data = []
        
        for token, data in search_process['token_postings'].items():
            token_data.append({
                'Term': token,
                'Document Frequency': data['df'],
                'IDF Value': data['idf']
            })
        
        if token_data:
            df_tokens = pd.DataFrame(token_data)
            
            # Bar chart of document frequencies
            fig_df = px.bar(
                df_tokens,
                x='Term',
                y='Document Frequency',
                title='Document Frequency of Query Terms',
                color='Document Frequency',
                color_continuous_scale='Blues'
            )
            figures['doc_freq'] = fig_df
            
            # Bar chart of IDF values
            fig_idf = px.bar(
                df_tokens,
                x='Term',
                y='IDF Value',
                title='IDF Values of Query Terms',
                color='IDF Value',
                color_continuous_scale='Viridis'
            )
            figures['idf'] = fig_idf
        
        # Create score contribution visualization
        if 'scoring_details' in search_process:
            score_data = []
            
            for doc_id, terms in search_process['scoring_details'].items():
                for term, details in terms.items():
                    score_data.append({
                        'Document': doc_id,
                        'Term': term,
                        'Score Contribution': details['contribution']
                    })
            
            if score_data:
                df_scores = pd.DataFrame(score_data)
                
                # Stacked bar chart of score contributions
                fig_scores = px.bar(
                    df_scores,
                    x='Document',
                    y='Score Contribution',
                    color='Term',
                    title='Score Contribution by Term for Each Document',
                    barmode='stack'
                )
                figures['scores'] = fig_scores
    
    # If this is a phrase search
    elif 'matching_details' in search_process:
        # Create match position visualization
        match_data = []
        
        for doc_id, details in search_process['matching_details'].items():
            match_data.append({
                'Document': doc_id,
                'Match Count': len(details['match_positions'])
            })
        
        if match_data:
            df_matches = pd.DataFrame(match_data)
            
            # Bar chart of match counts
            fig_matches = px.bar(
                df_matches,
                x='Document',
                y='Match Count',
                title='Phrase Match Counts by Document',
                color='Match Count',
                color_continuous_scale='Reds'
            )
            figures['matches'] = fig_matches
    
    return figures

def streamlit_corpus_explorer(docs, metadata=None):
    """
    Streamlit interface for exploring corpus characteristics
    
    Parameters:
    -----------
    docs : dict
        Dictionary of document texts
    metadata : dict, optional
        Dictionary of document metadata
    """
    st.subheader("Corpus Explorer")
    
    # Display basic corpus statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Documents", len(docs))
    
    with col2:
        total_words = sum(len(re.findall(r'\b\w+\b', text)) for text in docs.values())
        st.metric("Total Words", total_words)
    
    with col3:
        avg_length = sum(len(text) for text in docs.values()) / len(docs) if docs else 0
        st.metric("Average Document Length", f"{avg_length:.1f} chars")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Document Statistics", "Term Distribution", "Author Demographics"])
    
    with tab1:
        # Plot document statistics
        figures = plot_corpus_stats(docs, metadata)
        
        # Show document lengths
        if 'doc_lengths' in figures:
            st.plotly_chart(figures['doc_lengths'])
        
        # Show unique words
        if 'unique_words' in figures:
            st.plotly_chart(figures['unique_words'])
        
        # Show years if available
        if 'years' in figures:
            st.plotly_chart(figures['years'])
    
    with tab2:
        # Plot term frequency distribution
        top_n = st.slider("Number of top terms to show:", min_value=10, max_value=50, value=20)
        term_fig = plot_term_frequency_distribution(docs, top_n=top_n)
        st.plotly_chart(term_fig)
        
        # Compare cultural terms vs general terms
        st.subheader("Cultural vs. General Terms Comparison")
        
        # Define default term sets
        default_cultural = ["african", "american", "renaissance", "latinx", "indigenous"]
        default_general = ["life", "time", "people", "world", "history"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            cultural_input = st.text_area("Cultural terms (one per line):", "\n".join(default_cultural))
            cultural_terms = [term.strip() for term in cultural_input.split("\n") if term.strip()]
        
        with col2:
            general_input = st.text_area("General terms (one per line):", "\n".join(default_general))
            general_terms = [term.strip() for term in general_input.split("\n") if term.strip()]
        
        if cultural_terms and general_terms:
            term_figures = visualize_cultural_term_comparison(docs, cultural_terms, general_terms)
            
            # Show documents containing each term
            if 'doc_count' in term_figures:
                st.plotly_chart(term_figures['doc_count'])
            
            # Show total occurrences
            if 'occurrences' in term_figures:
                st.plotly_chart(term_figures['occurrences'])
            
            # Show averages
            if 'averages' in term_figures:
                st.plotly_chart(term_figures['averages'])
    
    with tab3:
        if metadata:
            # Try to extract author demographics
            author_genders = {}
            author_nationalities = {}
            
            for doc_id, meta in metadata.items():
                # Look for gender in metadata
                for key, value in meta.items():
                    if key.lower() in ('gender', 'author_gender', 'sex'):
                        if isinstance(value, str):
                            author_genders[doc_id] = value.strip().lower()
                    
                    if key.lower() in ('nationality', 'country', 'origin', 'author_nationality'):
                        if isinstance(value, str):
                            author_nationalities[doc_id] = value.strip()
            
            # Show demographics if available
            if author_genders:
                gender_counts = Counter(author_genders.values())
                fig_genders = px.pie(
                    names=list(gender_counts.keys()),
                    values=list(gender_counts.values()),
                    title='Author Gender Distribution',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_genders)
            
            if author_nationalities:
                nationality_counts = Counter(author_nationalities.values())
                # Sort by count for better visualization
                sorted_nationalities = sorted(nationality_counts.items(), key=lambda x: x[1], reverse=True)
                nationality_df = pd.DataFrame(sorted_nationalities, columns=['Nationality', 'Count'])
                
                fig_nationalities = px.bar(
                    nationality_df,
                    x='Nationality',
                    y='Count',
                    title='Author Nationality Distribution',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_nationalities)
            
            if not author_genders and not author_nationalities:
                st.info("No author demographic information found in metadata.")
        else:
            st.info("No metadata provided for demographic analysis.")
        
        # Show bias implications
        st.subheader("Corpus Representation Bias")
        
        st.markdown("""
        The composition of a corpus can significantly impact search results:
        
        1. **Historical bias**: Older texts may over-represent certain perspectives while excluding others. For example, Project Gutenberg's collection is heavily weighted toward Western European and American authors from certain time periods.
        
        2. **Gender and racial representation**: If authors of particular genders or ethnicities are under-represented, terms and topics important to those groups may be disadvantaged in search.
        
        3. **Language bias**: Non-English terms and concepts may be less frequent or missing entirely, affecting their retrieval.
        
        4. **Topic bias**: Certain topics may be over-represented (e.g., literature from specific genres) while others are under-represented.
        """)

def streamlit_preprocessing_demo():
    """
    Streamlit interface for exploring text preprocessing
    """
    st.title("Text Preprocessing: Where Bias Begins")
    
    st.markdown("""
    Before documents can be indexed and searched, they undergo preprocessing - a series of transformations
    that prepare them for efficient retrieval. While these steps are technical necessities, they can introduce
    bias in subtle ways.
    
    Let's explore how preprocessing affects different types of text.
    """)
    
    # Sample texts
    sample_texts = {
        "Standard English": "The quick brown fox jumps over the lazy dog. She was running toward the finish line.",
        "Names & Places": "María Rodríguez-López visited O'Connor's Pub in São Paulo while traveling from Việt Nam.",
        "Cultural Terms": "The hip-hop artist explored themes of diaspora, code-switching, and afrofuturism in her work.",
        "Mixed Language": "She felt that comforting sense of déjà vu as the mariachi band played a beautiful corrido."
    }
    
    # Text selection
    selected_text_type = st.selectbox(
        "Select text type to analyze:",
        list(sample_texts.keys())
    )
    
    # Custom text input
    use_custom = st.checkbox("Or enter your own text")
    
    if use_custom:
        text_input = st.text_area("Enter text to preprocess:", 
                               "Enter your text here...",
                               height=100)
    else:
        text_input = sample_texts[selected_text_type]
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        keep_case = st.checkbox("Preserve case", value=False)
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
    
    with col2:
        do_stemming = st.checkbox("Apply stemming", value=True)
    
    # Process text and display results
    if text_input and text_input != "Enter your text here...":
        # Get visualization data
        viz_data = visualize_preprocessing_steps(
            text_input,
            do_stemming=do_stemming,
            keep_case=keep_case,
            remove_stopwords=remove_stopwords
        )
        
        # Show original and final text
        st.subheader("Preprocessing Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Text:**")
            st.text(viz_data['original'])
        
        with col2:
            st.markdown("**Processed Text:**")
            st.text(viz_data['processed'])
        
        # Information loss statistics
        info_loss = viz_data['information_loss']
        
        st.subheader("Information Loss Analysis")
        
        loss_col1, loss_col2, loss_col3 = st.columns(3)
        
        with loss_col1:
            st.metric(
                "Token Reduction",
                f"{info_loss['token_reduction_percentage']:.1f}%",
                delta=-info_loss['token_reduction_percentage'],
                delta_color="inverse"
            )
        
        with loss_col2:
            st.metric(
                "Original Tokens",
                info_loss['original_token_count']
            )
        
        with loss_col3:
            st.metric(
                "Final Tokens",
                info_loss['final_token_count']
            )
        
        # Step-by-step visualization
        st.subheader("Step-by-Step Preprocessing")
        
        for i, step in enumerate(viz_data['steps'], 1):
            with st.expander(f"Step {i}: {step['name']}"):
                st.markdown(f"**Description:** {step['description']}")
                
                st.markdown("**Input:**")
                st.text(step['input'])
                
                st.markdown("**Output:**")
                st.text(step['output'])
                
                # Show bias implications
                st.markdown("**Potential Bias:**")
                st.info(step['bias_implications'])
                
                # Show highlighted differences if available
                if 'highlighted_diffs' in step and step['highlighted_diffs']:
                    st.markdown("**Examples of Information Loss:**")
                    
                    for diff in step['highlighted_diffs']:
                        st.markdown(f"- '{diff['original']}' → '{diff['processed']}'")
        
        # Special analysis for stemming
        if do_stemming:
            st.subheader("Stemming Analysis")
            
            # Extract terms for stemming visualization
            if use_custom:
                # Extract notable terms from custom text
                tokens = re.findall(r'\b[a-zA-Z]{4,}\b', text_input)
                unique_tokens = list(set(tokens))
                # Limit to 15 terms
                terms_to_analyze = unique_tokens[:15]
            else:
                # Use predefined terms for each sample text
                predefined_terms = {
                    "Standard English": ["jumps", "jumping", "jumped", "runner", "running", "ran", "quickly", "quickest", "laziness", "lazily"],
                    "Names & Places": ["Maria", "Rodriguez", "Lopez", "Sao", "Paulo", "Vietnam", "Vietnamese"],
                    "Cultural Terms": ["diaspora", "diasporic", "afrofuturism", "afrofuturist", "switching", "switched", "culture", "cultural"],
                    "Mixed Language": ["deja", "mariachi", "corrido", "corridos", "beautiful", "beautifully", "comforting", "comforted"]
                }
                terms_to_analyze = predefined_terms.get(selected_text_type, predefined_terms["Standard English"])
            
            # Create stemming visualization
            if terms_to_analyze:
                stemming_fig = visualize_stemming_effects(terms_to_analyze)
                st.plotly_chart(stemming_fig)
                
                # Show analysis of stemming bias
                st.markdown("""
                ### Stemming Bias Analysis
                
                Stemming algorithms like Porter's were designed primarily for English text and can introduce bias in several ways:
                
                1. **Cultural terms**: Terms specific to non-Western cultures may be stemmed incorrectly.
                2. **Names**: Names from different cultures may be incorrectly stemmed, losing important distinctions.
                3. **Loanwords**: Words borrowed from other languages often follow different morphological patterns.
                4. **Semantic nuance**: Stemming assumes words with the same root have related meanings, which isn't always true.
                """)
        
        # Case study for specific text types
        if not use_custom:
            st.subheader("Case Study Analysis")
            
            if selected_text_type == "Names & Places":
                st.markdown("""
                ### Impact on Names & Places
                
                Preprocessing can significantly affect how names and places are handled:
                
                - **Diacritics removal**: "María Rodríguez" becomes "maria rodriguez", losing cultural specificity
                - **Apostrophes**: "O'Connor" becomes "oconnor"
                - **Compound names**: "Rodríguez-López" becomes separate terms
                - **Non-Western places**: "Việt Nam" becomes "viet nam"
                
                These transformations can make names from certain cultures more difficult to search for or distinguish.
                """)
            
            elif selected_text_type == "Cultural Terms":
                st.markdown("""
                ### Impact on Cultural Terms
                
                Cultural terms often suffer from preprocessing:
                
                - **Compound terms**: "hip-hop" and "afrofuturism" may be split or stemmed incorrectly
                - **Culture-specific concepts**: Terms like "diaspora" or "code-switching" may be rare in the corpus
                - **Neologisms**: Newer cultural terms may not be well-represented in training data for stemmers
                
                These effects can make certain cultural concepts more difficult to search for effectively.
                """)
            
            elif selected_text_type == "Mixed Language":
                st.markdown("""
                ### Impact on Multilingual Text
                
                Text containing multiple languages faces particular challenges:
                
                - **Diacritics**: "déjà vu" loses its diacritics, which can change meaning in the original language
                - **Loanwords**: "mariachi" and "corrido" may be processed with English-centric algorithms
                - **Stopwords**: English stopword lists may inappropriately remove important words in other languages
                - **Stemming**: Applying English stemming rules to non-English words produces incorrect results
                
                These issues can make multilingual concepts harder to represent and retrieve accurately.
                """)

def visualize_ir_system():
    """
    Streamlit interface for visualizing the complete IR system
    """
    st.title("Information Retrieval System Visualization")
    
    st.markdown("""
    Information Retrieval (IR) systems consist of several interconnected components.
    Bias can enter the system at multiple points and compound through the pipeline.
    """)
    
    # Show IR flowchart
    st.subheader("IR System Architecture")
    
    # Toggle for bias highlighting
    show_bias = st.checkbox("Highlight bias entry points", value=True)
    
    # Generate Mermaid diagram code
    mermaid_code = visualize_bias_entry_points(highlights=show_bias)
    
    # Display mermaid diagram
    st.markdown(mermaid_code)
    
    # Explanation of bias entry points
    st.subheader("Bias Entry Points Explained")
    
    bias_points = {
        "Document Collection (Selection Bias)": 
            """The corpus itself may over-represent certain perspectives, time periods, or cultures
            while under-representing or excluding others. For example, Project Gutenberg contains
            primarily older Western works in the public domain.""",
        
        "Text Preprocessing (Stemming Bias)": 
            """Preprocessing steps like case folding, punctuation removal, and stemming can
            disproportionately affect certain languages, names, or cultural terms.""",
        
        "Build Inverted Index (Context Loss)": 
            """Converting documents to term-document mappings loses important context. Terms are
            treated as independent units, disconnected from their surrounding meaning.""",
        
        "Query Preprocessing (Same Stemming Bias)": 
            """The same preprocessing applied to documents is applied to queries, which can
            introduce similar biases and make certain queries more difficult to express.""",
        
        "Boolean Retrieval (Exact Match Bias)": 
            """Boolean retrieval requires exact matching of terms, which can disadvantage
            concepts that might be expressed in various ways across different documents.""",
        
        "Phrase Query Retrieval (Strict Order Bias)": 
            """Phrase queries require terms to appear in exactly the specified order, which
            may not accommodate different grammatical structures across languages.""",
        
        "TF-IDF Calculation (Statistical Bias)": 
            """Statistical methods like TF-IDF can amplify corpus biases by giving higher
            weight to terms that are rare in the corpus - which may include culturally
            specific terms simply because of corpus composition rather than actual importance.""",
        
        "Cosine Similarity (Vector Space Bias)": 
            """Vector space models treat documents as points in high-dimensional space,
            but this mathematical construct may not accurately represent semantic relationships
            between terms, especially across different cultural contexts."""
    }
    
    # Display bias points with expandable explanations
    for point, explanation in bias_points.items():
        with st.expander(point):
            st.markdown(explanation)
    
    # Mitigation strategies
    st.subheader("Bias Mitigation Strategies")
    
    st.markdown("""
    Several approaches can help reduce algorithmic bias in IR systems:
    
    1. **Diverse corpus selection**: Ensure document collections represent diverse perspectives, time periods, languages, and cultures.
    
    2. **Culture-aware preprocessing**: Develop preprocessing techniques that respect different naming conventions, language structures, and cultural terms.
    
    3. **Multilingual support**: Implement language detection and language-specific processing rather than applying English-centric techniques to all text.
    
    4. **Context preservation**: Use techniques that preserve contextual relationships between terms rather than treating them as independent units.
    
    5. **Evaluation with diverse queries**: Test retrieval systems with queries representing different cultural perspectives and information needs.
    
    6. **Transparent documentation**: Clearly document the limitations and potential biases in the system for users.
    """)

if __name__ == "__main__":
    # This allows the file to be run directly with streamlit
    st.set_page_config(page_title="IR Bias Visualization", layout="wide")
    
    # Main page navigation
    page = st.sidebar.radio(
        "Select Page",
        ["Corpus Explorer", "Text Preprocessing", "IR System Visualization"]
    )
    
    # Sample documents for demonstration
    sample_docs = {
        "pride_prejudice": """It is a truth universally acknowledged, that a single man in possession of a good fortune, 
        must be in want of a wife. However little known the feelings or views of such a man may be 
        on his first entering a neighbourhood, this truth is so well fixed in the minds of the 
        surrounding families, that he is considered the rightful property of some one or other of 
        their daughters.""",
        
        "harlem_renaissance": """The Harlem Renaissance was an intellectual and cultural revival of African American music,
        dance, art, fashion, literature, and politics. It fostered new styles of expression and
        influenced American culture broadly. Artists like Langston Hughes, Zora Neale Hurston, and
        Louis Armstrong defined this era.""",
        
        "scientific_text": """Scientists have discovered a new species of deep-sea creatures living near hydrothermal vents.
        These organisms thrive in extreme conditions without sunlight, using chemosynthesis instead
        of photosynthesis. This finding expands our understanding of how life can adapt to harsh
        environments.""",
        
        "cultural_terms": """Many AAVE (African American Vernacular English) terms have entered mainstream vocabulary.
        Similarly, words from Latinx communities around the world have enriched English.
        Code-switching between languages and dialects is common in multilingual communities."""
    }
    
    # Sample metadata
    sample_metadata = {
        "pride_prejudice": {
            "author": "Jane Austen",
            "year": 1813,
            "gender": "female",
            "nationality": "British"
        },
        "harlem_renaissance": {
            "author": "Various",
            "year": 1920,
            "nationality": "American"
        },
        "scientific_text": {
            "author": "Anonymous Researcher",
            "year": 2022,
            "genre": "Scientific"
        },
        "cultural_terms": {
            "author": "Contemporary Writer",
            "year": 2023,
            "genre": "Cultural Studies"
        }
    }
    
    # Show selected page
    if page == "Corpus Explorer":
        streamlit_corpus_explorer(sample_docs, sample_metadata)
    elif page == "Text Preprocessing":
        streamlit_preprocessing_demo()
    else:  # IR System Visualization
        visualize_ir_system()