# utils/system_bias.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import math
from nltk.stem import PorterStemmer
import nltk

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def run_bias_comparison_search(query, index):
    """
    Run a search with and without bias mitigation and compare results
    
    Parameters:
    -----------
    query : str
        Search query
    index : InvertedIndex
        Search index
    """
    st.markdown(f"### Search Results for: '{query}'")
    
    # Set up columns for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Standard Search (With Bias)")
        
        # Perform standard search
        results, process = index.search(query, top_k=5)
        
        if results:
            for i, (doc_id, score) in enumerate(results, 1):
                # Get metadata if available
                metadata = index.metadata.get(doc_id, {})
                
                # Format title
                title = metadata.get('title', doc_id)
                
                with st.expander(f"{i}. {title} (Score: {score:.4f})", expanded=True if i <= 3 else False):
                    # Format metadata nicely if available
                    if metadata:
                        st.write(f"**Author:** {metadata.get('author', 'Unknown')}")
                        st.write(f"**Year:** {metadata.get('year', 'Unknown')}")
                        st.write(f"**Author Gender:** {metadata.get('gender', 'Unknown')}")
                        st.write(f"**Nationality:** {metadata.get('nationality', 'Unknown')}")
                    
                    # Show document snippet
                    snippet = index.get_document_snippet(doc_id)
                    st.markdown(f"**Excerpt:**\n> {snippet}")
        else:
            st.write("No results found for this query.")
        
        # Show query processing details
        with st.expander("Query Processing Details"):
            st.markdown("**Original query:**")
            st.text(process['query'])
            
            st.markdown("**Processed query tokens:**")
            st.text(str(process['processed_query']))
            
            st.markdown("**Token statistics:**")
            for token, data in process.get('token_postings', {}).items():
                st.text(f"Token '{token}': Found in {data.get('df', 0)} documents, IDF = {data.get('idf', 0):.4f}")
        
    with col2:
        st.markdown("#### Bias-Mitigated Search")
        
        # Simulate a bias-mitigated search
        # In a real implementation, you would have actual mitigation code
        mitigated_results = simulate_mitigated_search(query, index)
        
        if mitigated_results:
            for i, result in enumerate(mitigated_results, 1):
                # Unpack result data
                doc_id = result['doc_id']
                score = result['score']
                explanation = result['explanation']
                
                # Get metadata if available
                metadata = index.metadata.get(doc_id, {})
                
                # Format title
                title = metadata.get('title', doc_id)
                
                with st.expander(f"{i}. {title} (Score: {score:.4f})", expanded=True if i <= 3 else False):
                    # Format metadata nicely if available
                    if metadata:
                        st.write(f"**Author:** {metadata.get('author', 'Unknown')}")
                        st.write(f"**Year:** {metadata.get('year', 'Unknown')}")
                        st.write(f"**Author Gender:** {metadata.get('gender', 'Unknown')}")
                        st.write(f"**Nationality:** {metadata.get('nationality', 'Unknown')}")
                    
                    # Show document snippet
                    snippet = index.get_document_snippet(doc_id)
                    st.markdown(f"**Excerpt:**\n> {snippet}")
                    
                    # Show mitigation explanation
                    st.markdown("**Mitigation applied:**")
                    st.info(explanation)
        else:
            st.write("No results found for this query.")
        
        # Show mitigation details
        with st.expander("Mitigation Details"):
            st.markdown("""
            **Bias mitigations applied:**
            
            1. **Corpus bias correction**: Adjusted document weights based on demographic representation
            2. **Preprocessing preservation**: Preserved case, diacritics, and compound terms
            3. **Context-aware matching**: Considered term context rather than isolated matches
            4. **TF-IDF recalibration**: Adjusted term weights to account for representation issues
            """)
    
    # Show comparison analysis
    st.subheader("Bias Impact Analysis")
    
    # Analyze the differences between standard and mitigated results
    analyze_search_difference(results, mitigated_results, query)


def simulate_mitigated_search(query, index):
    """
    Simulate a bias-mitigated search to demonstrate the concept
    
    In a real implementation, you would have actual mitigation code
    rather than this simulation.
    
    Parameters:
    -----------
    query : str
        Search query
    index : InvertedIndex
        Search index
        
    Returns:
    --------
    list
        Simulated mitigated search results
    """
    # Perform standard search as a base
    results, _ = index.search(query, top_k=10)
    
    # If no results, return empty list
    if not results:
        return []
    
    # Convert results to a list we can modify
    mitigated_results = []
    
    # Identify if the query contains potentially bias-affected terms
    # This is a simplified detection - in reality, you'd need more sophisticated methods
    query_lower = query.lower()
    potential_cultural_terms = ["indigenous", "african", "latinx", "hispanic", "diaspora", 
                                "afrofuturism", "non-western", "maría", "nguyễn", "o'connor"]
    
    has_cultural_terms = any(term in query_lower for term in potential_cultural_terms)
    has_diacritics = any(c in "áéíóúüñçãõàèìòùâêîôû" for c in query)
    has_punctuation = any(c in "-'" for c in query)
    
    # Simulate different mitigations based on query characteristics
    for doc_id, score in results:
        # Get metadata for this document
        metadata = index.metadata.get(doc_id, {})
        
        # Start with the original score
        mitigated_score = score
        explanation = "No specific mitigation needed for this result."
        
        # Simulate mitigation for cultural terms
        if has_cultural_terms:
            # Boost score for docs with diverse authorship
            nationality = metadata.get('nationality', '').lower()
            non_western_nationalities = ["african", "asian", "latin", "indigenous"]
            
            if any(nat in nationality for nat in non_western_nationalities):
                mitigated_score *= 1.25
                explanation = "Score boosted for cultural representation (diverse authorship)."
        
        # Simulate mitigation for diacritics and punctuation
        elif has_diacritics or has_punctuation:
            # These terms likely lost information in preprocessing
            mitigated_score *= 1.2
            explanation = "Score adjusted to compensate for information loss in preprocessing."
        
        # Simulate temporal bias correction
        publication_year = metadata.get('year', 0)
        try:
            # Convert to integer if it's a string
            if isinstance(publication_year, str) and publication_year.isdigit():
                publication_year = int(publication_year)
                
            # Adjust recent works slightly to balance historical bias
            if publication_year > 1950:
                mitigated_score *= 1.1
                explanation = "Score adjusted to balance historical bias in the corpus."
        except (ValueError, TypeError):
            pass
        
        # Simulate gender representation correction
        gender = metadata.get('gender', '').lower()
        if gender == 'female' or gender == 'non-binary':
            mitigated_score *= 1.15
            explanation = "Score adjusted to improve gender representation in results."
        
        # Add to mitigated results
        mitigated_results.append({
            'doc_id': doc_id,
            'score': mitigated_score,
            'explanation': explanation
        })
    
    # Sort by mitigated score
    mitigated_results = sorted(mitigated_results, key=lambda x: x['score'], reverse=True)
    
    # Take top 5
    return mitigated_results[:5]


def analyze_search_difference(standard_results, mitigated_results, query):
    """
    Analyze the differences between standard and mitigated search results
    
    Parameters:
    -----------
    standard_results : list
        Standard search results (doc_id, score)
    mitigated_results : list
        Mitigated search results (dict with doc_id, score, explanation)
    query : str
        Search query
    """
    # Check if both result sets exist
    if not standard_results or not mitigated_results:
        st.write("Insufficient results for comparison.")
        return
    
    # Get document IDs from each result set
    standard_doc_ids = [doc_id for doc_id, _ in standard_results]
    mitigated_doc_ids = [result['doc_id'] for result in mitigated_results]
    
    # Find differences in ranking
    rank_changes = {}
    for i, doc_id in enumerate(standard_doc_ids):
        if doc_id in mitigated_doc_ids:
            # Document appears in both result sets
            new_rank = mitigated_doc_ids.index(doc_id)
            rank_change = i - new_rank  # Positive means improved rank
            rank_changes[doc_id] = rank_change
    
    # Count documents that appear only in one result set
    only_in_standard = [doc_id for doc_id in standard_doc_ids if doc_id not in mitigated_doc_ids]
    only_in_mitigated = [result['doc_id'] for result in mitigated_results if result['doc_id'] not in standard_doc_ids]
    
    # Display analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Results Only in Standard", len(only_in_standard))
    
    with col2:
        st.metric("Results Only in Mitigated", len(only_in_mitigated))
    
    with col3:
        # Calculate average rank improvement for documents in both sets
        if rank_changes:
            avg_rank_change = sum(rank_changes.values()) / len(rank_changes)
            st.metric("Avg. Rank Change", f"{avg_rank_change:.2f}", 
                     delta=f"{avg_rank_change:.2f}", delta_color="normal")
        else:
            st.metric("Avg. Rank Change", "N/A")
    
    # Create visualization of rank changes
    if rank_changes:
        # Prepare data for bar chart
        change_data = []
        for doc_id, change in rank_changes.items():
            # Find the document in the mitigated results
            for result in mitigated_results:
                if result['doc_id'] == doc_id:
                    mitigated_score = result['score']
                    break
            else:
                mitigated_score = 0
            
            # Find standard score
            for std_doc_id, std_score in standard_results:
                if std_doc_id == doc_id:
                    standard_score = std_score
                    break
            else:
                standard_score = 0
            
            # Add to data
            change_data.append({
                'Document': doc_id,
                'Rank Change': change,
                'Standard Score': standard_score,
                'Mitigated Score': mitigated_score,
                'Score Difference': mitigated_score - standard_score
            })
        
        # Create dataframe
        change_df = pd.DataFrame(change_data)
        
        # Sort by rank change
        change_df = change_df.sort_values('Rank Change', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            change_df,
            x='Document',
            y='Rank Change',
            color='Score Difference',
            title='Document Rank Changes (Positive = Improved Rank)',
            labels={'Rank Change': 'Rank Improvement', 'Document': 'Document ID'},
            color_continuous_scale='RdBu',
            text='Rank Change'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Document",
            yaxis_title="Rank Change (positions)",
            coloraxis_colorbar_title="Score Diff."
        )
        
        # Display chart
        st.plotly_chart(fig)
    
    # Provide interpretation
    st.markdown("### Interpretation of Bias Impact")
    
    if only_in_mitigated:
        # Get details of new documents
        new_docs_analysis = []
        for doc_id in only_in_mitigated:
            # Find the document in mitigated results
            for result in mitigated_results:
                if result['doc_id'] == doc_id:
                    # Get metadata
                    metadata = index.metadata.get(doc_id, {})
                    
                    new_docs_analysis.append({
                        'doc_id': doc_id,
                        'title': metadata.get('title', doc_id),
                        'author': metadata.get('author', 'Unknown'),
                        'nationality': metadata.get('nationality', 'Unknown'),
                        'gender': metadata.get('gender', 'Unknown'),
                        'year': metadata.get('year', 'Unknown'),
                        'explanation': result['explanation']
                    })
                    break
        
        if new_docs_analysis:
            st.markdown("#### New Documents Added by Bias Mitigation")
            for i, doc in enumerate(new_docs_analysis, 1):
                st.markdown(f"{i}. **{doc['title']}** by {doc['author']} ({doc['year']})")
                st.markdown(f"   - {doc['nationality']} author, {doc['gender']}")
                st.markdown(f"   - Reason: {doc['explanation']}")
    
    # General interpretation
    query_lower = query.lower()
    cultural_query = any(term in query_lower for term in ["indigenous", "african", "latinx", "diaspora", "maría"])
    
    if cultural_query:
        st.info("""
        **Cultural Query Analysis**: This query contains culturally specific terms that may be affected by bias in the IR system.
        
        The standard search likely under-represents diverse perspectives due to:
        - Corpus selection bias (fewer works by diverse authors)
        - Preprocessing bias (loss of diacritics, compound terms)
        - TF-IDF bias (cultural terms have skewed statistics)
        
        The mitigated search attempts to correct these biases by:
        - Boosting underrepresented perspectives
        - Preserving important cultural markers
        - Adjusting term weights to account for representation issues
        """)
    else:
        st.info("""
        **General Query Analysis**: This query contains more general terms that may be less affected by obvious bias.
        
        However, even general queries can reflect hidden biases in the IR system:
        - Historical perspective bias (older works dominate the corpus)
        - Geographic and cultural centrism (Western perspectives may dominate)
        - Gender representation issues
        
        The mitigated search attempts to provide more balanced results across:
        - Time periods
        - Cultural perspectives
        - Gender and demographic representation
        """)


def trace_term_through_system(term, index):
    """
    Trace a term through each step of the IR system to show bias effects
    
    Parameters:
    -----------
    term : str
        Term to trace
    index : InvertedIndex
        Search index
    """
    st.markdown(f"### Tracing '{term}' Through the IR Pipeline")
    
    # Create a flowchart-like visualization of the term's journey
    steps = []
    
    # Step 1: Corpus representation
    corpus_stage = {
        "stage": "Corpus Representation",
        "description": "How well is this term represented in our document collection?",
        "input": term,
        "output": "",
        "bias_impact": ""
    }
    
    # Search for the exact term to check representation
    results, _ = index.search(term, top_k=100)
    doc_count = len(results)
    total_docs = len(index.documents)
    percentage = (doc_count / total_docs) * 100 if total_docs > 0 else 0
    
    corpus_stage["output"] = f"Found in {doc_count} out of {total_docs} documents ({percentage:.2f}%)"
    
    # Assess bias impact
    if percentage < 5:
        corpus_stage["bias_impact"] = "HIGH BIAS: Term is significantly underrepresented in the corpus"
    elif percentage < 15:
        corpus_stage["bias_impact"] = "MODERATE BIAS: Term has below-average representation"
    else:
        corpus_stage["bias_impact"] = "LOW BIAS: Term has adequate representation"
    
    steps.append(corpus_stage)
    
    # Step 2: Preprocessing
    # Simulate preprocessing steps
    import re
    from nltk.stem import PorterStemmer
    
    preprocessing_stage = {
        "stage": "Text Preprocessing",
        "description": "How is the term transformed during preprocessing?",
        "input": term,
        "output": "",
        "bias_impact": ""
    }
    
    # Apply preprocessing steps
    lowercase = term.lower()
    no_punct = re.sub(r'[^\w\s]', '', lowercase)
    
    # Apply stemming
    stemmer = PorterStemmer()
    stemmed = ' '.join([stemmer.stem(token) for token in no_punct.split()])
    
    preprocessing_stage["output"] = f"After preprocessing: '{stemmed}'"
    
    # Assess information loss
    changes = []
    if term != lowercase:
        changes.append("Case information lost")
    if lowercase != no_punct:
        changes.append("Punctuation removed")
    if no_punct != stemmed:
        changes.append("Word stems modified")
    
    if changes:
        preprocessing_stage["bias_impact"] = f"INFORMATION LOSS: {', '.join(changes)}"
    else:
        preprocessing_stage["bias_impact"] = "NO INFORMATION LOSS: Term preserved through preprocessing"
    
    steps.append(preprocessing_stage)
    
    # Step 3: Indexing
    indexing_stage = {
        "stage": "Indexing",
        "description": "How is the term stored in the inverted index?",
        "input": stemmed,
        "output": "",
        "bias_impact": ""
    }
    
    # Check if stemmed term exists in index
    stemmed_parts = stemmed.split()
    index_representation = []
    
    for part in stemmed_parts:
        if part in index.index:
            doc_count = len(index.index[part])
            index_representation.append(f"'{part}' found in {doc_count} documents")
        else:
            index_representation.append(f"'{part}' not found in index")
    
    indexing_stage["output"] = "Index entries: " + ", ".join(index_representation)
    
    # Assess bias impact
    if any("not found" in entry for entry in index_representation):
        indexing_stage["bias_impact"] = "HIGH BIAS: Some parts of the term are missing from the index"
    elif len(stemmed_parts) > 1:
        indexing_stage["bias_impact"] = "MODERATE BIAS: Compound term is split into separate components"
    else:
        indexing_stage["bias_impact"] = "LOW BIAS: Term is properly represented in the index"
    
    steps.append(indexing_stage)
    
    # Step 4: Query processing
    query_stage = {
        "stage": "Query Processing",
        "description": "How would this term be processed in a user query?",
        "input": term,
        "output": stemmed,
        "bias_impact": "Same preprocessing bias as in document indexing"
    }
    
    steps.append(query_stage)
    
    # Step 5: TF-IDF calculation
    tfidf_stage = {
        "stage": "TF-IDF Calculation",
        "description": "How does TF-IDF scoring affect this term?",
        "input": stemmed,
        "output": "",
        "bias_impact": ""
    }
    
    # Calculate IDF for stemmed parts
    idf_values = []
    
    for part in stemmed_parts:
        if part in index.index:
            # Get document frequency
            df = len(index.index[part])
            # Calculate IDF
            idf = math.log(total_docs / df) if df > 0 else 0
            idf_values.append(f"IDF('{part}') = {idf:.4f}")
        else:
            idf_values.append(f"IDF('{part}') = N/A (term not in index)")
    
    tfidf_stage["output"] = "TF-IDF values: " + ", ".join(idf_values)
    
    # Assess bias impact
    if any("N/A" in idf for idf in idf_values):
        tfidf_stage["bias_impact"] = "HIGH BIAS: Some terms are missing from the index, affecting TF-IDF calculation"
    elif any(part in stemmed_parts for part in ["african", "indigenous", "latinx", "diaspora"]):
        tfidf_stage["bias_impact"] = "POTENTIAL BIAS: Cultural terms may have artificially high IDF due to underrepresentation"
    elif len(stemmed_parts) > 1:
        tfidf_stage["bias_impact"] = "MODERATE BIAS: Compound term's meaning is distributed across multiple TF-IDF calculations"
    else:
        avg_idf = sum([float(idf.split("=")[1].strip()[:-1]) for idf in idf_values if "N/A" not in idf]) / len(idf_values) if idf_values else 0
        if avg_idf > 8:
            tfidf_stage["bias_impact"] = "POTENTIAL BIAS: Very high IDF indicates extreme rarity, possibly due to corpus bias"
        else:
            tfidf_stage["bias_impact"] = "LOW BIAS: Term has reasonable TF-IDF properties"
    
    steps.append(tfidf_stage)
    
    # Step 6: Final ranking
    ranking_stage = {
        "stage": "Result Ranking",
        "description": "How does this term affect final search result ranking?",
        "input": "TF-IDF scores and other ranking factors",
        "output": "",
        "bias_impact": ""
    }
    
    # Run a search to see where this term ranks
    results, _ = index.search(term, top_k=5)
    
    if results:
        ranking_output = []
        for i, (doc_id, score) in enumerate(results, 1):
            # Get document metadata
            metadata = index.metadata.get(doc_id, {})
            title = metadata.get('title', doc_id)
            author = metadata.get('author', 'Unknown')
            ranking_output.append(f"#{i}: {title} by {author} (Score: {score:.4f})")
        
        ranking_stage["output"] = "Top search results:\n" + "\n".join(ranking_output)
    else:
        ranking_stage["output"] = "No search results found for this term"
    
    # Assess bias impact
    if not results:
        ranking_stage["bias_impact"] = "EXTREME BIAS: Term produces no search results despite potentially being important"
    else:
        # Analyze diversity of top results
        authors = [index.metadata.get(doc_id, {}).get('author', 'Unknown') for doc_id, _ in results]
        genders = [index.metadata.get(doc_id, {}).get('gender', 'Unknown') for doc_id, _ in results]
        nationalities = [index.metadata.get(doc_id, {}).get('nationality', 'Unknown') for doc_id, _ in results]
        years = [index.metadata.get(doc_id, {}).get('year', 'Unknown') for doc_id, _ in results]
        
        unique_authors = len(set([a for a in authors if a != 'Unknown']))
        unique_genders = len(set([g for g in genders if g != 'Unknown']))
        unique_nationalities = len(set([n for n in nationalities if n != 'Unknown']))
        
        if unique_authors < 2 or unique_genders < 2 or unique_nationalities < 2:
            ranking_stage["bias_impact"] = "HIGH BIAS: Top results lack diversity in authorship, gender, or nationality"
        else:
            ranking_stage["bias_impact"] = "MODERATE BIAS: Results show some diversity but may still reflect corpus limitations"
    
    steps.append(ranking_stage)
    
    # Display the term trace
    for i, step in enumerate(steps, 1):
        st.markdown(f"#### Step {i}: {step['stage']}")
        
        col1, col2 = st.columns([3, 7])
        
        with col1:
            st.markdown("**Description:**")
            st.markdown(step['description'])
            
            st.markdown("**Input:**")
            st.markdown(f"`{step['input']}`")
        
        with col2:
            st.markdown("**Output:**")
            st.markdown(f"`{step['output']}`")
            
            st.markdown("**Bias Impact:**")
            
            # Format based on bias level
            bias_text = step['bias_impact']
            if "HIGH BIAS" in bias_text or "EXTREME BIAS" in bias_text:
                st.error(bias_text)
            elif "MODERATE BIAS" in bias_text or "POTENTIAL BIAS" in bias_text:
                st.warning(bias_text)
            else:
                st.success(bias_text)
    
    # Overall bias summary
    st.subheader("Overall Bias Analysis")
    
    # Count bias levels
    high_bias_count = sum(1 for step in steps if "HIGH BIAS" in step['bias_impact'] or "EXTREME BIAS" in step['bias_impact'])
    moderate_bias_count = sum(1 for step in steps if "MODERATE BIAS" in step['bias_impact'] or "POTENTIAL BIAS" in step['bias_impact'])
    low_bias_count = sum(1 for step in steps if "LOW BIAS" in step['bias_impact'])
    
    # Create a bias score (simple weighted sum)
    bias_score = (high_bias_count * 3 + moderate_bias_count * 1) / len(steps) * 10
    
    # Display bias meter
    st.markdown("**System-Level Bias Score:**")
    
    col1, col2, col3 = st.columns([1, 10, 1])
    
    with col2:
        # Create a progress bar representing bias
        st.progress(bias_score / 10)
        
        # Add labels
        label_col1, label_col2, label_col3 = st.columns(3)
        with label_col1:
            st.markdown("Low Bias")
        with label_col2:
            st.markdown("Moderate Bias")
        with label_col3:
            st.markdown("High Bias")
        
        # Display numeric score
        st.markdown(f"**Score: {bias_score:.1f}/10**")
    
    # Show bias explanation
    if bias_score < 3:
        st.success(f"""
        **Low System-Level Bias Detected**
        
        The term '{term}' experiences relatively little bias as it flows through the IR system. This suggests
        it is well-represented in the corpus and adequately handled by the search algorithms.
        """)
    elif bias_score < 7:
        st.warning(f"""
        **Moderate System-Level Bias Detected**
        
        The term '{term}' experiences some bias as it moves through the IR system. This bias compounds
        across multiple stages, potentially leading to suboptimal search results. Specific issues include:
        
        - {steps[0]['bias_impact']}
        - {steps[1]['bias_impact']}
        """)
    else:
        st.error(f"""
        **High System-Level Bias Detected**
        
        The term '{term}' experiences significant bias throughout the IR pipeline. This bias compounds
        across multiple stages, leading to severely compromised search results. Critical issues include:
        
        - {steps[0]['bias_impact']}
        - {steps[1]['bias_impact']}
        - {steps[2]['bias_impact']}
        """)


def run_mitigated_search(query, mitigations, index):
    """
    Run a search with selected bias mitigation strategies
    
    Parameters:
    -----------
    query : str
        Search query
    mitigations : list
        List of mitigation strategies to apply
    index : InvertedIndex
        Search index
    """
    st.markdown(f"### Mitigated Search Results for: '{query}'")
    
    # Create a description of applied mitigations
    mitigation_descriptions = {
        "preserve_case": "Preserved case information to protect proper nouns and cultural terms",
        "custom_stemming": "Applied cultural-aware stemming that preserves important distinctions",
        "context_aware": "Used context-aware indexing to maintain term relationships",
        "term_boosting": "Applied boosting for underrepresented cultural and specialized terms",
        "diverse_corpus": "Applied corpus diversity weighting to balance historical and cultural biases"
    }
    
    # Show applied mitigations
    if mitigations:
        st.markdown("**Applied Mitigations:**")
        for mitigation in mitigations:
            st.success(mitigation_descriptions.get(mitigation, mitigation))
    else:
        st.info("No mitigations selected. Showing standard search results.")
    
    # Perform search and simulate mitigation effects
    results, _ = index.search(query, top_k=10)
    
    # If no results, show message and return
    if not results:
        st.write("No results found for this query.")
        return
    
    # Apply selected mitigations (simulated)
    mitigated_results = []
    
    for doc_id, score in results:
        # Get metadata for this document
        metadata = index.metadata.get(doc_id, {})
        
        # Start with the original score
        mitigated_score = score
        applied_mitigations = []
        
        # Apply each selected mitigation
        for mitigation in mitigations:
            if mitigation == "preserve_case" and any(c.isupper() for c in query):
                # Simulate case preservation
                mitigated_score *= 1.1
                applied_mitigations.append("Case information preserved")
            
            elif mitigation == "custom_stemming" and any(term in query.lower() for term in ["diaspora", "indigenous", "latinx"]):
                # Simulate custom stemming for cultural terms
                mitigated_score *= 1.15
                applied_mitigations.append("Cultural-aware stemming applied")
            
            elif mitigation == "context_aware" and len(query.split()) > 1:
                # Simulate context-aware indexing for multi-word queries
                mitigated_score *= 1.2
                applied_mitigations.append("Context relationships maintained")
            
            elif mitigation == "term_boosting":
                # Simulate term boosting based on metadata
                nationality = metadata.get('nationality', '').lower()
                gender = metadata.get('gender', '').lower()
                
                if any(nat in nationality for nat in ["african", "asian", "latin"]):
                    mitigated_score *= 1.25
                    applied_mitigations.append("Diverse perspective boosted")
                
                if gender == "female" or gender == "non-binary":
                    mitigated_score *= 1.1
                    applied_mitigations.append("Gender diversity boosted")
            
            elif mitigation == "diverse_corpus":
                # Simulate corpus diversity weighting
                year = metadata.get('year', 0)
                try:
                    if isinstance(year, str) and year.isdigit():
                        year = int(year)
                    
                    if year > 1950:
                        mitigated_score *= 1.1
                        applied_mitigations.append("Recent work boosted to balance historical bias")
                except (ValueError, TypeError):
                    pass
        
        # Add to mitigated results
        mitigated_results.append({
            'doc_id': doc_id,
            'original_score': score,
            'mitigated_score': mitigated_score,
            'metadata': metadata,
            'applied_mitigations': applied_mitigations
        })
    
    # Sort by mitigated score
    mitigated_results = sorted(mitigated_results, key=lambda x: x['mitigated_score'], reverse=True)
    
    # Display results
    for i, result in enumerate(mitigated_results[:5], 1):
        doc_id = result['doc_id']
        metadata = result['metadata']
        title = metadata.get('title', doc_id)
        
        with st.expander(f"{i}. {title} (Score: {result['mitigated_score']:.4f})", expanded=True if i <= 3 else False):
            # Show metadata
            st.write(f"**Author:** {metadata.get('author', 'Unknown')}")
            st.write(f"**Year:** {metadata.get('year', 'Unknown')}")
            st.write(f"**Author Gender:** {metadata.get('gender', 'Unknown')}")
            st.write(f"**Nationality:** {metadata.get('nationality', 'Unknown')}")
            
            # Show score change
            score_change = result['mitigated_score'] - result['original_score']
            change_percent = (score_change / result['original_score']) * 100 if result['original_score'] > 0 else 0
            
            if score_change > 0:
                st.success(f"Score increased by {score_change:.4f} ({change_percent:.1f}%)")
            else:
                st.info(f"Score unchanged")
            
            # Show applied mitigations
            if result['applied_mitigations']:
                st.markdown("**Applied mitigations:**")
                for mitigation in result['applied_mitigations']:
                    st.markdown(f"- {mitigation}")
            
            # Show document snippet
            snippet = index.get_document_snippet(doc_id)
            st.markdown(f"**Excerpt:**\n> {snippet}")
    
    # Create visualization of score changes
    viz_data = []
    for result in mitigated_results[:5]:  # Top 5 results
        doc_id = result['doc_id']
        metadata = result['metadata']
        title = metadata.get('title', doc_id)
        
        viz_data.append({
            'Title': title if len(title) < 30 else title[:27] + "...",
            'Original Score': result['original_score'],
            'Mitigated Score': result['mitigated_score'],
            'Score Increase': result['mitigated_score'] - result['original_score'],
            'Percent Increase': ((result['mitigated_score'] / result['original_score']) - 1) * 100 if result['original_score'] > 0 else 0
        })
    
    # Create dataframe
    viz_df = pd.DataFrame(viz_data)
    
    # Create visualization
    fig = px.bar(
        viz_df,
        x='Title',
        y=['Original Score', 'Mitigated Score'],
        barmode='group',
        title='Score Comparison: Original vs. Mitigated',
        labels={'value': 'Score', 'variable': 'Score Type'},
        color_discrete_map={'Original Score': '#36A2EB', 'Mitigated Score': '#4CAF50'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Document",
        yaxis_title="Score",
        legend_title="Score Type"
    )
    
    # Display chart
    st.plotly_chart(fig)
    
    # Display mitigation explanation
    st.subheader("How Mitigations Address System-Level Bias")
    
    st.markdown("""
    The selected mitigation strategies work together to address bias at multiple levels of the IR system:
    
    1. **Corpus-level bias** is addressed by reweighting results to compensate for historical and demographic imbalances in the document collection.
    
    2. **Preprocessing bias** is mitigated by preserving case information and using culturally-aware stemming that respects important distinctions.
    
    3. **Indexing bias** is reduced through context-aware techniques that maintain relationships between terms rather than treating them as isolated units.
    
    4. **Statistical bias** in term weighting is countered by boosting underrepresented perspectives and terms.
    
    These approaches help create a more equitable search experience that better serves diverse information needs.
    """)


def run_system_bias_analysis(analysis_type, index):
    """
    Run a system-level bias analysis based on the selected type
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis to perform
    index : InvertedIndex
        Search index
    """
    st.markdown("### System-Level Bias Analysis")
    
    if analysis_type == "cultural_vs_general":
        # Cultural vs General terms analysis
        st.markdown("""
        #### Cultural Terms vs. General Terms Analysis
        
        This analysis compares how the search system handles culturally specific terms versus general terms.
        We'll examine corpus representation, preprocessing effects, and ranking behavior.
        """)
        
        # Define term sets
        cultural_terms = ["indigenous", "diaspora", "afrofuturism", "latinx", "african-american"]
        general_terms = ["science", "history", "literature", "society", "education"]
        
        with st.spinner("Analyzing term behavior..."):
            # Analyze both sets of terms
            term_data = []
            
            for term_type, terms in [("Cultural", cultural_terms), ("General", general_terms)]:
                for term in terms:
                    # Search for the term
                    results, process = index.search(term, top_k=10)
                    
                    # Get basic statistics
                    doc_count = len(results)
                    
                    # Get token info if available
                    if 'token_postings' in process and term.lower() in process['token_postings']:
                        token_info = process['token_postings'][term.lower()]
                        df = token_info.get('df', 0)
                        idf = token_info.get('idf', 0)
                    else:
                        df = 0
                        idf = 0
                    
                    # Check for preprocessing changes
                    import re
                    from nltk.stem import PorterStemmer
                    
                    lowercase = term.lower()
                    no_punct = re.sub(r'[^\w\s]', '', lowercase)
                    
                    stemmer = PorterStemmer()
                    stemmed = ' '.join([stemmer.stem(token) for token in no_punct.split()])
                    
                    # Calculate information loss
                    orig_tokens = term.split()
                    processed_tokens = stemmed.split()
                    token_ratio = len(processed_tokens) / len(orig_tokens) if orig_tokens else 1
                    
                    # Check for punctuation loss
                    punct_loss = 1 if re.search(r'[^\w\s]', term) and not re.search(r'[^\w\s]', stemmed) else 0
                    
                    # Check for case loss
                    case_loss = 1 if re.search(r'[A-Z]', term) and not re.search(r'[A-Z]', stemmed) else 0
                    
                    # Add to data
                    term_data.append({
                        'Term': term,
                        'Type': term_type,
                        'Document Count': df,
                        'IDF Value': idf,
                        'Processed Form': stemmed,
                        'Token Ratio': token_ratio,
                        'Punctuation Loss': punct_loss,
                        'Case Loss': case_loss,
                        'Information Loss Score': punct_loss + case_loss + (1 - token_ratio if token_ratio < 1 else 0)
                    })
            
            # Create dataframe
            term_df = pd.DataFrame(term_data)
            
            # Create visualizations
            
            # IDF comparison
            fig1 = px.box(
                term_df,
                x='Type',
                y='IDF Value',
                title='IDF Values by Term Type',
                color='Type',
                points='all'
            )
            st.plotly_chart(fig1)
            
            # Information loss comparison
            fig2 = px.box(
                term_df,
                x='Type',
                y='Information Loss Score',
                title='Information Loss in Preprocessing by Term Type',
                color='Type',
                points='all'
            )
            st.plotly_chart(fig2)
            
            # Document representation
            fig3 = px.bar(
                term_df,
                x='Term',
                y='Document Count',
                color='Type',
                title='Document Count by Term Type',
                barmode='group'
            )
            st.plotly_chart(fig3)
            
            # Show detailed data table
            st.subheader("Detailed Term Analysis")
            st.dataframe(term_df)
            
            # Calculate averages by type
            type_averages = term_df.groupby('Type').agg({
                'Document Count': 'mean',
                'IDF Value': 'mean',
                'Information Loss Score': 'mean'
            }).reset_index()
            
            # Show summary statistics
            st.subheader("Summary Statistics")
            st.table(type_averages)
            
            # Interpret results
            st.subheader("Interpretation")
            
            # Get average values
            cultural_avg_idf = type_averages[type_averages['Type'] == 'Cultural']['IDF Value'].values[0]
            general_avg_idf = type_averages[type_averages['Type'] == 'General']['IDF Value'].values[0]
            
            cultural_avg_docs = type_averages[type_averages['Type'] == 'Cultural']['Document Count'].values[0]
            general_avg_docs = type_averages[type_averages['Type'] == 'General']['Document Count'].values[0]
            
            cultural_avg_loss = type_averages[type_averages['Type'] == 'Cultural']['Information Loss Score'].values[0]
            general_avg_loss = type_averages[type_averages['Type'] == 'General']['Information Loss Score'].values[0]
            
            # Create interpretation
            st.markdown(f"""
            #### System-Level Bias Analysis Summary
            
            This analysis reveals several key patterns of bias in the IR system:
            
            1. **Corpus Representation Bias**:
               - Cultural terms appear in an average of {cultural_avg_docs:.1f} documents
               - General terms appear in an average of {general_avg_docs:.1f} documents
               - Cultural terms are {(general_avg_docs/cultural_avg_docs):.1f}x less represented in the corpus
            
            2. **Statistical Bias**:
               - Cultural terms have an average IDF of {cultural_avg_idf:.2f}
               - General terms have an average IDF of {general_avg_idf:.2f}
               - Cultural terms have {(cultural_avg_idf/general_avg_idf):.1f}x higher IDF values
            
            3. **Preprocessing Bias**:
               - Cultural terms experience {cultural_avg_loss:.2f} average information loss during preprocessing
               - General terms experience {general_avg_loss:.2f} average information loss
               - Cultural terms lose {(cultural_avg_loss/general_avg_loss):.1f}x more information
            
            #### System-Level Impact
            
            These biases compound through the IR pipeline:
            
            1. Cultural terms start with less representation in the corpus
            2. They lose more information during preprocessing
            3. They receive artificial statistical importance due to their rarity
            4. The result is a system that may retrieve fewer, potentially less relevant documents for cultural queries
            
            This demonstrates how individual biases at different stages combine to create significant system-level bias.
            """)
    
    elif analysis_type == "western_vs_nonwestern":
        # Western vs non-Western concepts analysis
        st.markdown("""
        #### Western vs. Non-Western Concepts Analysis
        
        This analysis examines how the search system handles Western vs. non-Western concepts.
        We'll compare representation, retrievability, and result diversity.
        """)
        
        # Define concept sets
        western_concepts = ["democracy", "enlightenment", "industrial revolution", "renaissance", "western philosophy"]
        nonwestern_concepts = ["ubuntu", "confucianism", "dharma", "indigenous knowledge", "non-western philosophy"]
        
        # Placeholder for analysis
        st.info("Analysis of Western vs. Non-Western concepts would be implemented here with similar methodology to the Cultural vs. General terms analysis.")
    
    elif analysis_type == "english_vs_noneng":
        # English vs non-English terms analysis
        st.markdown("""
        #### English vs. Non-English Terms Analysis
        
        This analysis compares how the search system handles English vs. non-English terms.
        We'll examine preprocessing impact, retrieval effectiveness, and bias patterns.
        """)
        
        # Define term sets
        english_terms = ["knowledge", "beautiful", "community", "history", "education"]
        noneng_terms = ["ubuntu", "hygge", "déjà vu", "zeitgeist", "wabi-sabi"]
        
        # Placeholder for analysis
        st.info("Analysis of English vs. Non-English terms would be implemented here with similar methodology to the Cultural vs. General terms analysis.")
    
    elif analysis_type == "gender_analysis":
        # Gender representation analysis
        st.markdown("""
        #### Gender Representation Analysis
        
        This analysis examines gender bias in search results across different query types.
        We'll analyze author gender distribution, topic coverage, and result diversity.
        """)
        
        # Define gender-related queries
        queries = ["women writers", "female authors", "gender studies", "feminism", "masculinity"]
        
        # Placeholder for analysis
        st.info("Gender representation analysis would be implemented here, examining author gender distributions in search results for different query types.")
