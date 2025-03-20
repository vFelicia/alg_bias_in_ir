# run it
# PS C:\GitHubRepos\alg_bias_in_ir> python -m venv venv                     ---> create venv
# PS C:\GitHubRepos\alg_bias_in_ir> venv/Scripts/activate                   ---> activate venv
# (venv) PS C:\GitHubRepos\alg_bias_in_ir> pip install -r requirements.txt  ---> install packages in venv
# (venv) PS C:\GitHubRepos\alg_bias_in_ir> streamlit run app.py             ---> run the explorable

import streamlit as st

# Set page config FIRST before any other Streamlit commands
st.set_page_config(
    page_title="Search Engines and Hidden Biases",
    page_icon="🔍",
    layout="wide"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import graphviz
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
import pickle
import time

# import custom utils
from utils.preprocessing import preprocess_text
from utils.indexing import InvertedIndex
from utils.retrieval import search_documents
from utils.visualization import plot_corpus_stats
# Import the ir_visualization utilities
from utils.ir_visualizations import display_ir_system_visualization
# import for system-level bias
from utils.system_bias import (
    run_bias_comparison_search, 
    trace_term_through_system, 
    run_mitigated_search, 
    run_system_bias_analysis
)

# Navigation callback function
def navigate_to(page):
    st.session_state.page = page
    st.experimental_rerun()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def build_and_save_index(data_dir, index_path, metadata_path=None):
    """Build the index and save it to disk, including metadata if available"""
    # Load documents
    documents = {}
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_id = filename.replace('.txt', '')
                    documents[doc_id] = f.read()
            except Exception as e:
                print(f"Error reading file {filename}: {str(e)}")
    
    # Load metadata if available
    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            metadata_df = pd.read_csv(metadata_path)
            print(f"Metadata columns: {metadata_df.columns.tolist()}")
            
            # Process each row in the metadata
            for _, row in metadata_df.iterrows():
                # Convert Doc ID to string for matching
                if 'Doc ID' in row:
                    doc_id_str = str(row['Doc ID']).strip()
                    
                    # For each document, find if any filename contains this ID
                    for existing_doc_id in documents.keys():
                        # If the existing document ID contains the metadata ID
                        if doc_id_str in existing_doc_id:
                            metadata[existing_doc_id] = {
                                'title': row.get('Title', 'Unknown'),
                                'author': row.get('Author', 'Unknown'),
                                'year': row.get('Year Published', 'Unknown'),
                                'gender': row.get('Gender', 'Unknown'),
                                'nationality': row.get('Nationality', 'Unknown'),
                                'genre': row.get('Genre', 'Unknown')
                            }
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create a modified version of InvertedIndex that doesn't use lambda in defaultdict
    # This modification ensures it can be pickled
    index = InvertedIndex(use_custom_dict=True)
    
    # Add documents and their metadata to the index
    for doc_id, text in documents.items():
        doc_metadata = metadata.get(doc_id, None)
        index.add_document(doc_id, text, doc_metadata)
    
    # Save index to disk
    try:
        with open(index_path, 'wb') as f:
            pickle.dump(index, f)
        print(f"Successfully saved index to {index_path}")
    except Exception as e:
        print(f"Error saving index: {str(e)}")
        import traceback
        traceback.print_exc()

    # Debugging section
    print(f"Documents loaded: {len(documents)}")
    print(f"Sample document IDs: {list(documents.keys())[:5]}")
    if 'metadata_df' in locals():
        print(f"Metadata loaded: {len(metadata_df)}")
        print(f"Metadata sample: {metadata_df.head()}")
    print(f"Final metadata mapping: {len(metadata)}")
    
    return index

def load_or_build_index(data_dir, index_path, metadata_path=None, force_rebuild=False):
    """Load the index from disk or build it if necessary"""
    # Check if index file exists and is newer than the data directory and metadata
    if os.path.exists(index_path) and not force_rebuild:
        try:
            # Check data files
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
            
            # Get modification times
            index_mod_time = os.path.getmtime(index_path)
            data_mod_times = [os.path.getmtime(os.path.join(data_dir, f)) for f in data_files]
            metadata_mod_time = os.path.getmtime(metadata_path) if metadata_path and os.path.exists(metadata_path) else 0
            
            # Check if index is newer than all data sources
            if data_files and index_mod_time > max(data_mod_times) and index_mod_time > metadata_mod_time:
                # Index is up to date, load it
                try:
                    with open(index_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Error loading index: {str(e)}")
        except Exception as e:
            print(f"Error checking data directory: {str(e)}")
    
    # Index doesn't exist or is outdated, build it
    return build_and_save_index(data_dir, index_path, metadata_path)

# Main app code
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Introduction"
    
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), "data", "gutenberg")
    index_path = os.path.join(os.path.dirname(__file__), "data", "index.pickle")
    metadata_path = os.path.join(os.path.dirname(__file__), "data", "metadata.csv")
    
    # Add force rebuild option for debugging
    force_rebuild = st.sidebar.checkbox("Force rebuild index", value=False)
    
    # Load or build the index at app startup
    with st.spinner("Loading search index..."):
        global search_index  # Make it global so all pages can access it
        search_index = load_or_build_index(data_dir, index_path, metadata_path, force_rebuild=force_rebuild)
    
    # Debug information about metadata
    st.sidebar.write(f"Loaded {len(search_index.documents)} documents")
    if hasattr(search_index, 'metadata') and search_index.metadata:
        st.sidebar.write(f"Found metadata for {len(search_index.metadata)} documents")
    else:
        st.sidebar.write("")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Introduction", "Selection Bias and the Corpus", "Text Preprocessing Bias", 
         "Statistical Bias: TF-IDFs", "Putting It All Together", "Final Reflection"],
        index=["Introduction", "Selection Bias and the Corpus", "Text Preprocessing Bias", 
               "Statistical Bias: TF-IDFs", "Putting It All Together", "Final Reflection"].index(st.session_state.page)
    )
    
    # Update the session state when sidebar selection changes
    if page != st.session_state.page:
        st.session_state.page = page
        st.experimental_rerun()
    
    # Main content based on selected page
    if page == "Introduction":
        show_introduction()
    elif page == "Selection Bias and the Corpus":
        show_corpus_analysis()
    elif page == "Text Preprocessing Bias":
        show_preprocessing()
    elif page == "Statistical Bias: TF-IDFs":
        show_tfidf_calculator()
    elif page == "Putting It All Together":
        show_system_bias()
    elif page == "Final Reflection":
        show_reflection()

def show_introduction():
    st.title("Search Engines and Hidden Biases")
    
    st.markdown("""
    How many times have you used a search engine today? Search engines have become an essential part of our daily lives,
    helping us find information, answer questions, and discover new ideas. But have you ever wondered how these systems
    work behind the scenes, and what biases might be hidden in their algorithms?
    
    In this interactive essay, we'll explore the inner workings of Information Retrieval (IR) systems - the technology
    that powers search engines - and examine how algorithmic bias can affect search results.
    """)
    
    # Show the simplified IR system diagram first
    st.subheader("Understanding Information Retrieval Systems")
    
    st.markdown("""
    Before we dive in, let's take a look at the basic structure of an Information Retrieval system. 
    This simplified diagram shows the main components and where bias can enter the system:
    """)
    
    # Allow users to toggle between different views
    visualization_type = st.radio(
        "Visualization Type:",
        ["Simplified IR System", "Bias Points Focus", "Detailed IR System"],
        horizontal=True,
        index=0  # Default to simplified view
    )
    
    if visualization_type == "Simplified IR System":
        display_ir_system_visualization("simplified")
    elif visualization_type == "Bias Points Focus":
        display_ir_system_visualization("bias_points")
    else:
        display_ir_system_visualization("detailed")
    
    st.markdown("""
    Throughout this interactive essay, we'll explore each component of this system and how bias can enter 
    and compound at different stages. We'll focus on three main types of bias:
    
    1. **Selection Bias**: How the composition of the document collection affects results
    2. **Preprocessing Bias**: How text processing can lose important cultural information
    3. **Statistical Bias**: How even "unbiased" mathematical functions can amplify existing biases
    
    Our dataset for this interactive essay is a collection of sixty randomly sampled books sourced from Project Gutenberg.
    We kept track of the book's title, author, year published, gender of author, nationality of author, and book genre.
    """)
    
    # Display some stats about the loaded corpus
    st.info(f"Loaded {len(search_index.documents)} documents into the search index.")
    
    # Basic search interface
    st.subheader("Let's get you acquainted with our data and our search engine! Try a basic search of a word or phrase. It should return the title(s) of a book most related to that your word or phrase.")
    query = st.text_input("Enter your search query:")
    
    if query:
        # Now passing the index to the search function
        results, search_process = search_documents(query, index=search_index, num_results=5)
        
        st.write("Search results:")
        if results:
            for i, (doc_id, score) in enumerate(results, 1):
                # Get metadata if available
                metadata = search_index.metadata.get(doc_id, {})
                
                # Create an expander for each result
                with st.expander(f"{i}. **{doc_id}** (Score: {score:.4f})", expanded=True):
                    # Format metadata nicely if available
                    if metadata:
                        # Create two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Title:** {metadata.get('title', 'Unknown')}")
                            st.write(f"**Author:** {metadata.get('author', 'Unknown')}")
                            st.write(f"**Year:** {metadata.get('year', 'Unknown')}")
                        
                        with col2:
                            st.write(f"**Author Gender:** {metadata.get('gender', 'Unknown')}")
                            st.write(f"**Nationality:** {metadata.get('nationality', 'Unknown')}")
                            st.write(f"**Genre:** {metadata.get('genre', 'Unknown')}")
                    
                    # Show document snippet
                    snippet = search_index.get_document_snippet(doc_id)
                    st.markdown(f"**Excerpt:**\n> {snippet}")
        else:
            st.write("No results found for your query.")
    
    # Reflection box
    st.subheader("Reflection")
    st.markdown("""
    **What patterns do you notice in the results?**
    
    Think about the types of books that appeared in your search results. Were they what you expected?
    What might influence which books are ranked higher?
    """)
    st.text_area("Your reflections:", height=150)

        # Overview of the interactive essay
    st.subheader("How This Interactive Essay Works")
    
    st.markdown("""
    ### Learning Journey
    
    Throughout this interactive experience, you'll explore the different components of an information retrieval (IR) system and discover how algorithmic bias can enter and compound at each stage. Here's what to expect:
    
    **1. Selection Bias and the Corpus**
    - Learn and understand Selection Bias through exploratory data analysis of text
    - Analyze our Project Gutenberg dataset
    - Explore how the composition of a document collection influences search results
    - Investigate representation of different cultural terms and demographics in the dataset
    
    **2. Text Preprocessing Bias**
    - Learn and understand Preprocessing Bias through a text preprocesisng simulator
    - Experiment with a text preprocessing simulator
    - See how stemming, case folding, and punctuation removal affect different types of text
    - Discover how cultural terms, names, and non-English words can lose important information during preprocesisng
    
    **3. Statistical Bias: TF-IDF**
    - Learn and understand Statistical Bias by seeing how an "unbiased" math formula can amplify biases
    - Learn, broadly, how TF-IDFs work and produces weighed terms by documents
    - Calculate scores for different types of terms
    - Understand how rare terms can be overweighted regardless of actual importance
    
    **4. System-Level Bias**
    - Experience how biases compound through the entire system
    - Compare standard and bias-mitigated search results
    - Trace terms through each stage of the IR pipeline
    
    **5. Final Reflection**
    - Consider how we might design more equitable IR systems
    - Apply what you've learned to real-world examples
    
    ### How to Navigate
    
    You can use the sidebar to navigate between sections. While the content is designed to be explored sequentially, feel free to jump to any section that interests you.
    
    Each section includes:
    - Key concepts and explanations
    - Interactive components to experiment with
    - Reflection prompts to deepen your understanding
    - Visualizations to illustrate important principles
    - A summary of the key concepts at the end, with proposals for how to mitigate bias
    
    Let's begin by exploring how the composition of our document collection affects search results!
    """)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        st.button("Continue to 'Selection Bias' →", 
                  on_click=navigate_to, 
                  args=("Selection Bias and the Corpus",),
                  key="intro_next")

def show_corpus_analysis():
    st.title("The Corpus Matters: Project Gutenberg as a Case Study")
    
    # Explanation of corpus
    st.markdown("""
    For our exploration,
    we're using texts from Project Gutenberg, a digital library of free eBooks. In a formal Information Retrieval system,
    we would call our collection of books "The Corpus" - that is, a collection of documents. In our case, 
    our books are the documents within the corpus.
    
    Project Gutenberg primarily contains older, public domain works. Let's examine the characteristics of this collection
    and discuss how the composition of our corpus might influence search results.
    """)
    
    # Directly load the metadata file
    metadata_path = os.path.join(os.path.dirname(__file__), "data", "gutenberg", "metadata.csv")
    
    # Load metadata from CSV
    metadata_df = pd.read_csv(metadata_path)
    
    # Display a sample of the metadata for debugging (can be removed in production)
    with st.expander("View metadata sample"):
        st.dataframe(metadata_df.head())
        st.write(f"Columns available: {', '.join(metadata_df.columns)}")
    
    # Display corpus size info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Documents", len(metadata_df))
    
    with col2:
        if 'Year Published' in metadata_df.columns:
            year_range = f"{metadata_df['Year Published'].min()} - {metadata_df['Year Published'].max()}"
            st.metric("Publication Years", year_range)
        else:
            st.metric("Publication Years", "N/A")
    
    with col3:
        if 'Author' in metadata_df.columns:
            st.metric("Unique Authors", metadata_df['Author'].nunique())
        else:
            st.metric("Unique Authors", "N/A")
    
    # Visualization of corpus statistics
    st.subheader("Corpus Statistics")
    
    tab1, tab2, tab3 = st.tabs(["Publication Dates", "Author Demographics", "Geographic Distribution"])
    
    with tab1:
        if 'Year Published' in metadata_df.columns:
            # Convert to numeric, handling potential errors
            metadata_df['Year Published'] = pd.to_numeric(metadata_df['Year Published'], errors='coerce')
            
            # Create year distribution histogram
            fig = px.histogram(
                metadata_df.dropna(subset=['Year Published']), 
                x='Year Published',
                nbins=20,
                title='Distribution of Publication Years',
                labels={'Year Published': 'Publication Year', 'count': 'Number of Works'},
                color_discrete_sequence=['#3366CC']
            )
            fig.update_layout(xaxis_title="Publication Year", yaxis_title="Number of Works")
            st.plotly_chart(fig)
            
            # Add explanation
            st.markdown("""
            This histogram shows the distribution of publication years in our corpus. Note that Project Gutenberg
            primarily contains works that are in the public domain, which means they tend to be older.
            
            **Potential Bias Impact**: A corpus heavily weighted toward older publications may under-represent 
            contemporary concepts, modern terminology, and recent cultural perspectives.
            """)
        else:
            st.error("Publication year data is missing from metadata.")
            
    with tab2:
        if 'Gender' in metadata_df.columns:
            # Create gender distribution pie chart
            gender_counts = metadata_df['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig = px.pie(
                gender_counts, 
                values='Count', 
                names='Gender',
                title='Author Gender Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig)
            
            # Add explanation
            st.markdown("""
            This chart shows the gender distribution of authors in our corpus.
            
            **Potential Bias Impact**: If the corpus is heavily skewed toward one gender,
            themes, perspectives, and language more common to that gender may be over-represented
            in search results.
            """)
        else:
            st.error("Author gender data is missing from metadata.")
            
    with tab3:
        if 'Nationality' in metadata_df.columns:
            # Create nationality distribution bar chart
            nationality_counts = metadata_df['Nationality'].value_counts().reset_index()
            nationality_counts.columns = ['Nationality', 'Count']
            
            # Sort by count for better visualization
            nationality_counts = nationality_counts.sort_values('Count', ascending=False)
            
            fig = px.bar(
                nationality_counts,
                x='Nationality',
                y='Count',
                title='Author Nationality Distribution',
                color='Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig)
            
            # Add explanation
            st.markdown("""
            This chart shows the distribution of author nationalities in our corpus.
            
            **Potential Bias Impact**: Geographic skew can result in cultural bias. If the corpus is
            dominated by authors from certain regions, cultural references, language styles, and
            perspectives from those regions will be over-represented in search results.
            """)
        elif 'Genre' in metadata_df.columns:
            # Alternative: show genre distribution if nationality is not available
            genre_counts = metadata_df['Genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            
            # Sort by count for better visualization
            genre_counts = genre_counts.sort_values('Count', ascending=False)
            
            fig = px.bar(
                genre_counts,
                x='Genre',
                y='Count',
                title='Genre Distribution',
                color='Count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig)
            
            st.markdown("""
            This chart shows the distribution of genres in our corpus.
            
            **Potential Bias Impact**: If certain genres are over-represented, terminology and
            concepts common in those genres will have higher prominence in search results.
            """)
        else:
            st.error("Author nationality and genre data are missing from metadata.")

    # Broader impact of corpus diversity
    st.subheader("Impact of Corpus Diversity on Search Results")

    st.markdown("""
    ### How Corpus Composition Shapes Search Results

    The composition of our document collection fundamentally shapes what users can find and how different topics are represented in search results. Our analysis of Project Gutenberg reveals several important patterns:

    #### 1. Historical and Temporal Bias

    As we saw in the publication year distribution, our corpus heavily favors older works from specific time periods. This creates:
    - Over-representation of historical perspectives and terminology
    - Under-representation of contemporary concepts and language
    - Difficulty finding modern ideas expressed in their current form

    #### 2. Cultural and Geographic Representation

    The nationality distribution shows a clear skew toward certain regions and cultures:
    - Western perspectives and references dominate the corpus
    - Non-Western cultural concepts may be rare or missing entirely
    - Cultural terms from underrepresented groups receive artificially high distinctiveness scores

    #### 3. Gender and Identity Perspectives

    The gender distribution of authors affects which voices and viewpoints are amplified:
    - Language patterns and topics more common to the dominant gender may be over-represented
    - Concepts and terminology important to underrepresented genders may be harder to find
    - Search algorithms may learn to prioritize dominant perspectives as "more relevant"

    #### 4. Genre and Domain Skew

    The distribution of genres affects which domains of knowledge are well-represented:
    - Literary language may be over-represented compared to scientific or technical terminology
    - Certain subject matters receive more coverage than others
    - Domain-specific language from underrepresented fields may be treated as unusual or less relevant
    """)

    # Add interactive reflection element
    st.markdown("""
    ### Thought Experiment: Consider a Search for...

    Think about how corpus composition might affect searches for each of these topics:
    """)

    topic_tabs = st.tabs(["Technology", "Social Justice", "Global Perspectives"])

    with topic_tabs[0]:
        st.markdown("""
        **Searching for Modern Technology Concepts**
        
        Terms like "smartphone," "internet," or "artificial intelligence" in their modern usage:
        - Would rarely appear in historical texts
        - Might have different meanings in older contexts (e.g., "artificial intelligence" in early sci-fi)
        - Could return results that are conceptually unrelated to the modern meaning
        
        **Impact**: Users searching for modern technology concepts would likely receive sparse, outdated, or irrelevant results.
        """)

    with topic_tabs[1]:
        st.markdown("""
        **Searching for Contemporary Social Justice Concepts**
        
        Terms like "intersectionality," "microaggression," or "gender non-binary":
        - Would be absent from most historical texts
        - Modern frameworks for discussing identity and equality would be missing
        - Related but outdated terminology might dominate results
        
        **Impact**: Users exploring contemporary social issues would find limited relevant content, potentially reinforcing the impression that these concepts are "new" or "niche" rather than extensions of long-standing concerns.
        """)

    with topic_tabs[2]:
        st.markdown("""
        **Searching for Non-Western Cultural Concepts**
        
        Terms from non-Western cultural traditions or philosophies:
        - Would be less frequent and often filtered through Western perspectives
        - Might appear primarily in anthropological or colonial contexts rather than authentic voices
        - Could be exoticized or described with outdated terminology
        
        **Impact**: Users seeking diverse cultural perspectives would encounter a narrower, potentially biased representation filtered through dominant cultural lenses.
        """)
    
    # Hypothesis box
    st.subheader("Form a Hypothesis")
    st.markdown("""
    **How might this collection affect search results for modern concepts?**
    
    Based on the corpus characteristics you observed above, what concerns might emerge when searching for:
    - Modern technology terms
    - Contemporary social issues
    - Non-Western cultural concepts
    """)
    
    hypothesis = st.text_area("Your hypothesis:", height=150)
    
    # Educational summary
    st.header("What We've Learned: Corpus Bias in Information Retrieval")
    
    st.markdown("""
    In this section, we've explored how the composition of our document collection (Project Gutenberg texts) shapes search results. You've seen:
    
    - **Publication year distribution** that heavily favors older works, potentially under-representing contemporary concepts
    - **Author demographics** that may skew toward certain genders and nationalities
    - **Term distribution analysis** showing how cultural and identity terms appear less frequently than general terms
    
    These patterns reveal several types of bias in our corpus:
    
    1. **Historical bias**: Older texts over-represent perspectives from their time periods while excluding more recent viewpoints
    
    2. **Gender and racial representation**: Our analysis showed potential imbalances in author demographics, which means terms and topics important to under-represented groups may be disadvantaged in search
    
    3. **Language bias**: Non-English terms and concepts appear less frequently or may be missing entirely
    
    4. **Topic bias**: Certain genres and subject matters are over-represented while others receive less coverage
    
    5. **Temporal bias**: Contemporary concepts may be absent or expressed differently in older texts
    
    ### Mitigation Strategies
    
    - **Diverse corpus selection**: Ensure document collections represent diverse perspectives, time periods, and cultures
    - **Weighted indexing**: Adjust term weights based on corpus demographics to compensate for under-representation
    - **Contextual expansion**: Add related terms to queries to capture concepts that might be expressed differently across time periods
    - **Transparent documentation**: Clearly document the limitations and potential biases in the corpus for users
    """)

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("← Back to Introduction", 
                  on_click=navigate_to, 
                  args=("Introduction",),
                  key="corpus_prev")
    with col3:
        st.button("Continue to 'Preprocessing' →", 
                  on_click=navigate_to, 
                  args=("Text Preprocessing Bias",),
                  key="corpus_next")

def show_preprocessing():
    st.title("Text Preprocessing")
    
    st.markdown("""
    Before texts can be searched, they undergo preprocessing - a series of transformations that prepare them for efficient indexing.
    While these steps are technical necessities, they can introduce bias in subtle ways.
    
    Let's explore common preprocessing steps and see how they might affect different types of text using the Text Preprocessing Simulator.
    """)
    
    # Interactive preprocessing simulator
    st.subheader("Text Preprocessing Simulator")

    st.markdown("""
    ### How to use this simulator:

    1. **Select a text sample** from the dropdown or enter your own
    2. **Configure preprocessing options** by toggling the checkboxes
    3. **Observe each transformation step** and how the text changes
    4. **Examine the bias implications** in the expandable sections
    5. **Compare the original and final text** to see what information was lost

    Pay attention to how names with diacritics (signs on the letters), cultural terms, and non-English words are transformed, 
    and reflect on how these changes might affect search accuracy for different user groups.
    """)
    
    # Sample texts
    sample_texts = {
        "Standard English": "The quick brown fox jumps over the lazy dog. Isn't natural language processing AMAZING?",
        "Names & Places": "María Rodríguez-López visited O'Connor's Pub in São Paulo while traveling from Việt Nam.",
        "Cultural Terms": "The hip-hop artist explored themes of diaspora, code-switching, and afrofuturism in her work.",
        "Mixed Language": "She felt that comforting sense of déjà vu as the mariachi band played a beautiful corrido."
    }
    
    # Text selection
    selected_text_type = st.selectbox(
        "Select example text type:",
        list(sample_texts.keys())
    )
    
    # Custom text input
    use_custom = st.checkbox("Or enter your own text")
    
    if use_custom:
        input_text = st.text_area("Enter some text to preprocess:", 
                                height=100,
                                value="Enter your text here...")
        if input_text == "Enter your text here...":
            input_text = sample_texts[selected_text_type]
    else:
        input_text = sample_texts[selected_text_type]
    
    # Preprocessing options
    st.subheader("Preprocessing Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lowercase = st.checkbox("Convert to lowercase", value=True)
    
    with col2:
        remove_punctuation = st.checkbox("Remove punctuation", value=True)
    
    with col3:
        do_stemming = st.checkbox("Apply stemming", value=True)
    
    remove_stopwords = st.checkbox("Remove stopwords", value=True)
    
    if input_text:
        # Apply preprocessing step by step
        st.subheader("Step-by-Step Preprocessing")
        
        # Step 1: Original text
        st.markdown("**Original Text:**")
        st.text(input_text)
        
        # Step 2: Lowercase (optional)
        if lowercase:
            lowercase_text = input_text.lower()
            st.markdown("**After Lowercase Conversion:**")
            st.text(lowercase_text)
            current_text = lowercase_text
            
            # Highlight differences
            with st.expander("Bias implications of lowercase conversion"):
                st.markdown("""
                **Bias implications:**
                - Loss of proper noun distinction (names, places, organizations)
                - Cultural markers in capitalization patterns may be lost
                - Acronyms become indistinguishable from regular words
                
                **Examples of information loss in this text:**
                """)
                
                # Find capitalized words in original text
                import re
                capitals = re.findall(r'\b[A-Z][a-zA-Z]*\b', input_text)
                
                if capitals:
                    for word in capitals:
                        st.markdown(f"- '{word}' → '{word.lower()}'")
                else:
                    st.markdown("No capitalized words found in this text.")
        else:
            current_text = input_text
        
        # Step 3: Punctuation removal (optional)
        if remove_punctuation:
            import re
            no_punct_text = re.sub(r'[^\w\s]', '', current_text)
            st.markdown("**After Punctuation Removal:**")
            st.text(no_punct_text)
            current_text = no_punct_text
            
            # Highlight differences
            with st.expander("Bias implications of punctuation removal"):
                st.markdown("""
                **Bias implications:**
                - Apostrophes in names (O'Connor, D'Angelo) are lost
                - Hyphens in compound terms (African-American, code-switching) are removed
                - Diacritical marks in non-English words are stripped
                - Context from quotes, questions, and exclamations is lost
                
                **Examples of information loss in this text:**
                """)
                
                # Find punctuation in original text
                import re
                if lowercase:
                    source_text = input_text.lower()
                else:
                    source_text = input_text
                    
                punct_pattern = r'[^\w\s]'
                punct_matches = re.finditer(punct_pattern, source_text)
                
                examples_shown = 0
                for match in punct_matches:
                    punct = match.group(0)
                    start = max(0, match.start() - 10)
                    end = min(len(source_text), match.end() + 10)
                    context = source_text[start:end]
                    context_clean = re.sub(punct_pattern, '', context)
                    
                    st.markdown(f"- '{context}' → '{context_clean}'")
                    examples_shown += 1
                    if examples_shown >= 5:  # Limit to 5 examples
                        break
                        
                if examples_shown == 0:
                    st.markdown("No punctuation found in this text.")
        
        # Step 4: Tokenization
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        tokens = nltk.word_tokenize(current_text)
        st.markdown("**After Tokenization:**")
        st.text(str(tokens))
        
        # Step 5: Stopword removal (optional)
        if remove_stopwords:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
                
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
            
            filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
            st.markdown("**After Stopword Removal:**")
            st.text(str(filtered_tokens))
            current_tokens = filtered_tokens
            
            # Highlight differences
            with st.expander("Bias implications of stopword removal"):
                st.markdown("""
                **Bias implications:**
                - Some stopwords carry meaning in specific cultural contexts
                - English stopword lists may inappropriately apply to multilingual text
                - Certain speech patterns or dialects may use stopwords differently
                
                **Removed stopwords from this text:**
                """)
                
                removed = [token for token in tokens if token.lower() in stop_words]
                if removed:
                    st.markdown(", ".join([f"'{word}'" for word in removed]))
                else:
                    st.markdown("No stopwords found in this text.")
        else:
            current_tokens = tokens
        
        # Step 6: Stemming (optional)
        if do_stemming:
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            
            stemmed_tokens = [ps.stem(token) for token in current_tokens]
            st.markdown("**After Stemming:**")
            st.text(str(stemmed_tokens))
            
            # Create a table showing the stemming changes
            stemming_changes = [(original, stemmed) for original, stemmed in zip(current_tokens, stemmed_tokens) if original != stemmed]
            
            if stemming_changes:
                st.markdown("**Stemming Changes:**")
                stemming_df = pd.DataFrame(stemming_changes, columns=['Original', 'Stemmed'])
                st.table(stemming_df)
                
                # Highlight differences
                with st.expander("Bias implications of stemming"):
                    st.markdown("""
                    **Bias implications:**
                    - Porter stemmer was designed for English and may incorrectly stem words from other languages
                    - Cultural terms and proper nouns may be improperly stemmed
                    - Different inflections with distinct meanings may be conflated
                    - Specialized terminology may lose important distinctions
                    
                    **Examples of potentially problematic stemming:**
                    """)
                    
                    # Check for potential cultural terms or proper nouns that were stemmed
                    for orig, stemmed in stemming_changes:
                        if (orig[0].isupper() or 
                            any(term in orig.lower() for term in ['culture', 'language', 'ethnic', 'tradition', 'community', 'identity'])):
                            st.markdown(f"- '{orig}' → '{stemmed}' (potential cultural term or proper noun)")
            else:
                st.info("No stemming changes to display for this text.")
            
            final_text = " ".join(stemmed_tokens)
        else:
            final_text = " ".join(current_tokens)
        
        # Final result
        st.subheader("Final Processed Text")
        st.text(final_text)
        
        # Information loss statistics
        original_word_count = len(input_text.split())
        processed_word_count = len(final_text.split())
        token_reduction = (1 - processed_word_count / original_word_count) * 100 if original_word_count > 0 else 0
        
        st.subheader("Information Loss Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Word Count", original_word_count)
        
        with col2:
            st.metric("Processed Word Count", processed_word_count)
        
        with col3:
            st.metric("Token Reduction", f"{token_reduction:.1f}%")
    
    # Case studies
    st.subheader("Case Study: Impact on Cultural Terms and Names")
    
    case_study_tabs = st.tabs(["Names", "Cultural Terms", "Non-English Words"])
    
    with case_study_tabs[0]:
        st.markdown("""
        ### Impact on Names
        
        Names from different cultures are processed differently by standard preprocessing techniques:
        
        | Original Name | After Preprocessing | Information Lost |
        |---------------|---------------------|------------------|
        | María Rodríguez-López | maria rodriguez lopez | Diacritics, capitals, hyphen |
        | O'Connor | oconnor | Apostrophe, capitalization |
        | Nguyễn | nguyen | Diacritics, capitalization |
        | DeAndre | deandr | Capitalization pattern, word ending |
        
        **Bias Impact**: Names from non-English cultures often lose more information during preprocessing,
        making them harder to search for accurately. This creates an uneven playing field where some names
        are more searchable than others.
        """)
    
    with case_study_tabs[1]:
        st.markdown("""
        ### Impact on Cultural Terms
        
        Cultural and identity terms are often compound words or specialized vocabulary:
        
        | Original Term | After Preprocessing | Information Lost |
        |---------------|---------------------|------------------|
        | African-American | african american | Hyphenation, compound meaning |
        | Latinx | latinx | (preserved but rare in older corpora) |
        | code-switching | code switch | Hyphenation, inflection |
        | hip-hop | hip hop | Hyphenation, compound meaning |
        | afrofuturism | afrofutur | Word ending with conceptual meaning |
        
        **Bias Impact**: Cultural terms often undergo more substantial transformation during preprocessing,
        potentially leading to poorer retrievability. Terms specific to marginalized groups may be particularly affected.
        """)
    
    with case_study_tabs[2]:
        st.markdown("""
        ### Impact on Non-English Words
        
        Words from languages other than English face particular challenges:
        
        | Original Word | After Preprocessing | Information Lost |
        |---------------|---------------------|------------------|
        | déjà vu | deja vu | Diacritics, compound nature |
        | café | cafe | Diacritics |
        | niño | nino | Diacritics with phonetic meaning |
        | São Paulo | sao paulo | Diacritics, capitalization |
        | corrido (Spanish song) | corrido | Cultural context, might be stemmed incorrectly |
        
        **Bias Impact**: Non-English words lose diacritical marks that may be essential to their meaning.
        English-centric stemming algorithms may incorrectly transform words from other languages.
        """)
    
    # Reflection box
    st.subheader("Reflection")
    st.markdown("""
    **What information is lost in preprocessing?**
    
    Based on the examples above, what types of terms might be disadvantaged by standard preprocessing steps?
    How might this affect search results for different user groups?
    """)
    
    reflection = st.text_area("Your reflections on preprocessing bias:", height=150)
    
    # Educational summary
    st.header("What We've Learned: Text Preprocessing and Bias")
    
    st.markdown("""
    In this section, you've experimented with how preprocessing transforms text before it can be indexed and searched. You've observed:
    
    - How **lowercase conversion** erases distinctions between proper nouns and common words
    - How **punctuation removal** affects hyphenated terms, names with apostrophes, and words with diacritics
    - How **stopword removal** might eliminate words that carry meaning in specific cultural contexts
    - How **stemming** can incorrectly transform cultural terms and words from non-English languages
    
    Through the case studies, you've seen how these transformations disproportionately affect:
    
    - **Names from different cultures** (María Rodríguez-López → maria rodriguez lopez)
    - **Cultural and identity terms** (African-American → african american)
    - **Non-English words** (déjà vu → deja vu)
    
    ### Strategies to Mitigate Preprocessing Bias
    
    Several approaches can help reduce bias introduced during preprocessing:
    
    1. **Language-specific preprocessing**: Apply different preprocessing rules based on detected language
    
    2. **Preserving case information**: Store both original and lowercase versions to maintain proper nouns
    
    3. **Careful stemming**: Use more conservative stemming algorithms or lemmatization that respects morphological 
    differences across languages
    
    4. **Entity recognition**: Apply named entity recognition to preserve personal names, locations, and organizations
    
    5. **Compound word handling**: Preserve or specially index compound terms, especially culturally significant ones
    
    6. **Multilingual stopword lists**: Use language-appropriate stopword lists for multilingual corpora
    
    7. **Diacritic preservation**: Maintain diacritical marks or index both versions (with and without diacritics)
    
    8. **Context-aware preprocessing**: Consider document context when determining how to process special terms
    """)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("← Back to 'Selection Bias'", 
                  on_click=navigate_to, 
                  args=("Selection Bias and the Corpus",),
                  key="preproc_prev")
    with col3:
        st.button("Continue to 'TF-IDF' →", 
                  on_click=navigate_to, 
                  args=("Statistical Bias: TF-IDFs",),
                  key="preproc_next")

def show_tfidf_calculator():
    st.title("Understanding TF-IDF: Numbers That Shape Results")
    
    st.markdown("""
    Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used to reflect how characteristic a word is to a document in a collection.
    In our case, the TF-IDF score of a word will tell you how characteritic that word is to a particular book in our dataset! For example,
    inputting "Fitzwilliam Darcy" as a query should return to you Jane Austen's Pride and Prejudice, that book ranked highly compared to other books.
    
    TF-IDFs are one of the fundamental algorithms used in search engines to rank results. Let's explore how TF-IDF works and how it might introduce bias.
    """)
    
    # Explanation of TF-IDF
    with st.expander("How does TF-IDF work?"):
        st.markdown("""
        ### Term Frequency (TF)
        
        Term Frequency measures how frequently a term occurs in a document, typically as a raw count:
        
        $TF(t, d) = \\frac{\\text{Number of times term t appears in document d}}{\\text{Total number of terms in document d}}$
        
        ### Inverse Document Frequency (IDF)
        
        Inverse Document Frequency measures how important a term is across all documents:
        
        $IDF(t) = \\log\\left(\\frac{\\text{Total number of documents}}{\\text{Number of documents containing term t}}\\right)$
        
        ### TF-IDF Score
        
        The TF-IDF score is the product of TF and IDF:
        
        $TFIDF(t, d) = TF(t, d) \\times IDF(t)$
        
        This formula gives:
        - Higher weight to terms that appear frequently in a specific document
        - Lower weight to terms that appear in many documents (considered less distinctive)
        """)
    
    # TF-IDF Calculator
    st.subheader("TF-IDF Calculator")
    st.markdown("""
    In this interactive calculator, you'll explore how TF-IDF works by:
    1. Seeing which books in our collection contain specific terms
    2. Understanding why certain books rank higher than others
    3. Discovering how cultural and specialized terms might be treated differently than common words

    ### How to Use This Calculator:

    **Step 1:** Enter a word in the search box below (try "love", "time", or "indigenous")

    **Step 2:** Examine the results to see:
    - How many books contain this term
    - Which books rank highest for this term
    - The breakdown of the TF-IDF calculation

    **Step 3:** Try comparing different types of terms using the comparison tool below the calculator

    Through these experiments, you'll discover how seemingly "neutral" statistics can introduce bias into search results based on what's represented (or underrepresented) in the document collection.
    """)
    
    # Word input
    word = st.text_input("Enter a word to analyze:", value="love")
    
    if word:
        # Calculate TF-IDF using the search index
        with st.spinner("Calculating TF-IDF scores across the corpus..."):
            # Get documents containing the term
            results, search_process = search_index.search(word, top_k=10)
            
            if not results:
                st.warning(f"The term '{word}' was not found in any documents.")
            else:
                # Extract document IDs and scores
                docs_with_scores = {doc_id: score for doc_id, score in results}
                
                # Get the token postings info
                if 'token_postings' in search_process and word.lower() in search_process['token_postings']:
                    token_info = search_process['token_postings'][word.lower()]
                    
                    # Display term statistics
                    st.subheader(f"Statistics for term: '{word}'")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Document Frequency", token_info['df'])
                    
                    with col2:
                        df_percentage = (token_info['df'] / len(search_index.documents)) * 100
                        st.metric("Percentage of Corpus", f"{df_percentage:.2f}%")
                    
                    with col3:
                        st.metric("IDF Value", f"{token_info['idf']:.4f}")
                    
                    # Create a dataframe with TF-IDF details for each document
                    results_data = []
                    
                    for doc_id, score in results:
                        # Get document metadata if available
                        metadata = search_index.metadata.get(doc_id, {})
                        
                        # Get scoring details for this term in this document
                        if ('scoring_details' in search_process and 
                            doc_id in search_process['scoring_details'] and
                            word.lower() in search_process['scoring_details'][doc_id]):
                            
                            details = search_process['scoring_details'][doc_id][word.lower()]
                            
                            # Add result to data
                            results_data.append({
                                'Document': doc_id,
                                'Title': metadata.get('title', doc_id),
                                'Author': metadata.get('author', 'Unknown'),
                                'Year': metadata.get('year', 'Unknown'),
                                'TF': details['tf'],
                                'Score Contribution': details['contribution'],
                                'Total Score': score
                            })
                        else:
                            # Fallback if detailed scoring not available
                            results_data.append({
                                'Document': doc_id,
                                'Title': metadata.get('title', doc_id),
                                'Author': metadata.get('author', 'Unknown'),
                                'Year': metadata.get('year', 'Unknown'),
                                'TF': 'N/A',
                                'Score Contribution': 'N/A',
                                'Total Score': score
                            })
                    
                    # Create a dataframe and display as table
                    results_df = pd.DataFrame(results_data)
                    st.subheader("Top Documents by TF-IDF Score")
                    st.dataframe(results_df)
                    
                    # Create bar chart visualization
                    fig = px.bar(
                        results_df,
                        x='Document',
                        y='Total Score',
                        title=f"TF-IDF Scores for '{word}' Across Top Documents",
                        color='Total Score',
                        hover_data=['Author', 'Year'],
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(xaxis_title="Document", yaxis_title="TF-IDF Score")
                    st.plotly_chart(fig)
                    
                    # Show breakdown of how score is calculated for first document
                    if results_data:
                        with st.expander("How is the score calculated?"):
                            first_doc = results_data[0]
                            doc_id = first_doc['Document']
                            
                            st.markdown(f"### Score calculation for '{word}' in document '{first_doc['Title']}'")
                            
                            # Get term frequency
                            tf = first_doc['TF'] if first_doc['TF'] != 'N/A' else 0
                            st.markdown(f"""
                            **Step 1: Calculate Term Frequency (TF)**
                            
                            TF measures how frequently the term appears in this specific document, normalized by document length.
                            
                            TF('{word}', '{doc_id}') = {tf:.6f}
                            """)
                            
                            # Get IDF
                            idf = token_info['idf']
                            st.markdown(f"""
                            **Step 2: Use Inverse Document Frequency (IDF)**
                            
                            IDF measures how rare the term is across all documents.
                            
                            IDF('{word}') = {idf:.6f}
                            """)
                            
                            # Calculate TF-IDF
                            tfidf = first_doc['Score Contribution'] if first_doc['Score Contribution'] != 'N/A' else 0
                            st.markdown(f"""
                            **Step 3: Calculate TF-IDF Score**
                            
                            TF-IDF is the product of TF and IDF.
                            
                            TF-IDF('{word}', '{doc_id}') = {tf:.6f} × {idf:.6f} = {tfidf:.6f}
                            """)
                            
                            st.info("This document may have scores from other terms if your search included multiple words.")
                else:
                    st.warning("Detailed term information not available.")
    
    # Compare multiple terms
    st.subheader("Compare Different Types of Terms")

    st.markdown("""
    Now let's compare how TF-IDF treats different kinds of terms. This tool lets you see the statistical differences between:

    - **Cultural/Specialized Terms**: Words that relate to specific cultural identities, traditions, or specialized knowledge
    - **General/Common Terms**: Everyday words that appear frequently across many types of texts

    Try using the default terms or entering your own to see:
    - Which terms appear in more documents
    - How IDF values differ between term types
    - What this might mean for search relevance

    After running the comparison, examine the charts and analysis to understand how statistical differences might create bias in search results.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        cultural_terms_input = st.text_area("Cultural/Specialized Terms (one per line):", 
                                          "african\nindigenous\nfeminism\nqueer\nlatin")
        cultural_terms = [term.strip() for term in cultural_terms_input.split("\n") if term.strip()]
    
    with col2:
        general_terms_input = st.text_area("General/Common Terms (one per line):", 
                                         "love\ntime\nlife\nday\nworld")
        general_terms = [term.strip() for term in general_terms_input.split("\n") if term.strip()]
    
    if st.button("Compare Terms"):
        with st.spinner("Comparing terms across the corpus..."):
            # Process all terms
            all_terms = cultural_terms + general_terms
            term_types = ["Cultural/Specialized"] * len(cultural_terms) + ["General/Common"] * len(general_terms)

            comparison_data = []

            for term, term_type in zip(all_terms, term_types):
                # Search for the term
                results, search_process = search_index.search(term, top_k=5)
                
                # Get token postings info if available
                if ('token_postings' in search_process and 
                    term.lower() in search_process['token_postings']):
                    
                    token_info = search_process['token_postings'][term.lower()]
                    df = token_info['df']
                    idf = token_info['idf']
                    
                    # Add to comparison data
                    comparison_data.append({
                        'Term': term,
                        'Type': term_type,
                        'Document Frequency': df,
                        'Percentage of Corpus': (df / len(search_index.documents)) * 100,
                        'IDF Value': idf,
                        'Number of Results': len(results)
                    })
                else:
                    # Term not found
                    comparison_data.append({
                        'Term': term,
                        'Type': term_type,
                        'Document Frequency': 0,
                        'Percentage of Corpus': 0,
                        'IDF Value': 0,
                        'Number of Results': 0
                    })
            
            if comparison_data:
                # Create dataframe
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display table
                st.subheader("Term Comparison")
                st.dataframe(comparison_df)
                
                # Create visualization of document frequency
                fig1 = px.bar(
                    comparison_df,
                    x='Term',
                    y='Document Frequency',
                    color='Type',
                    title='Document Frequency by Term Type',
                    labels={'Document Frequency': 'Number of Documents'},
                    color_discrete_map={'Cultural/Specialized': '#FF6B6B', 'General/Common': '#4ECDC4'}
                )
                st.plotly_chart(fig1)
                
                # Create visualization of IDF values
                fig2 = px.bar(
                    comparison_df,
                    x='Term',
                    y='IDF Value',
                    color='Type',
                    title='IDF Values by Term Type',
                    labels={'IDF Value': 'Inverse Document Frequency'},
                    color_discrete_map={'Cultural/Specialized': '#FF6B6B', 'General/Common': '#4ECDC4'}
                )
                st.plotly_chart(fig2)
                
                # Calculate and display averages by type
                # Only calculate means for numeric columns
                numeric_cols = ['Document Frequency', 'Percentage of Corpus', 'IDF Value', 'Number of Results']
                avg_by_type = comparison_df.groupby('Type')[numeric_cols].mean().reset_index()
                
                st.subheader("Average Values by Term Type")
                st.dataframe(avg_by_type)
                
                # Create a summary chart
                summary_data = []
                for term_type in ['Cultural/Specialized', 'General/Common']:
                    type_data = comparison_df[comparison_df['Type'] == term_type]
                    summary_data.append({
                        'Type': term_type,
                        'Avg Document Frequency': type_data['Document Frequency'].mean(),
                        'Avg Percentage of Corpus': type_data['Percentage of Corpus'].mean(),
                        'Avg IDF Value': type_data['IDF Value'].mean()
                    })

                summary_df = pd.DataFrame(summary_data)

                # Instead of using multiple y values, reshape the dataframe first
                summary_melted = pd.melt(
                    summary_df, 
                    id_vars=['Type'],
                    value_vars=['Avg Document Frequency', 'Avg IDF Value'],
                    var_name='Metric',
                    value_name='Value'
                )

                # Then create the bar chart
                fig3 = px.bar(
                    summary_melted,
                    x='Type',
                    y='Value',
                    color='Metric',
                    barmode='group',
                    title='Average Statistics by Term Type',
                    color_discrete_sequence=['#FF9F1C', '#2EC4B6']
                )
                st.plotly_chart(fig3)
                
                # Analysis of results
                if avg_by_type.shape[0] == 2:  # If we have both types
                    cultural_avg_df = avg_by_type[avg_by_type['Type'] == 'Cultural/Specialized']['Document Frequency'].values[0]
                    general_avg_df = avg_by_type[avg_by_type['Type'] == 'General/Common']['Document Frequency'].values[0]
                    cultural_avg_idf = avg_by_type[avg_by_type['Type'] == 'Cultural/Specialized']['IDF Value'].values[0]
                    general_avg_idf = avg_by_type[avg_by_type['Type'] == 'General/Common']['IDF Value'].values[0]
                    
                    df_diff = general_avg_df - cultural_avg_df
                    idf_diff = cultural_avg_idf - general_avg_idf
                    
                    st.info(f"""
                    **Analysis:**
                    
                    - General terms appear in an average of {general_avg_df:.1f} documents, while cultural/specialized terms appear in {cultural_avg_df:.1f} documents.
                    - This means general terms are found in {df_diff:.1f} more documents on average ({(df_diff/cultural_avg_df*100):.1f}% more).
                    
                    - Cultural/specialized terms have an average IDF of {cultural_avg_idf:.4f}, compared to {general_avg_idf:.4f} for general terms.
                    - This means cultural terms have {idf_diff:.4f} higher IDF values on average ({(idf_diff/general_avg_idf*100):.1f}% higher).
                    
                    - Higher IDF values for cultural terms suggest they are treated as "more distinctive" by the algorithm.
                    - However, this could also reflect under-representation of these terms in the corpus rather than true information value.
                    """)
    
    # Hypothesis box
    st.subheader("Form a Hypothesis")
    st.markdown("""
    **How might TF-IDF scores differ for cultural terms vs. general terms?**
    
    Try entering different types of words in the calculator above:
    - Common English words (e.g., "love", "time", "day")
    - Cultural terms (e.g., "diaspora", "indigenous")
    - Technical terms from different fields
    
    What patterns do you notice? What biases might this introduce in search results?
    """)
    
    hypothesis = st.text_area("Your hypothesis about TF-IDF bias:", height=150)
    
    # Educational context
    st.header("What We've Learned: How TF-IDF Can Introduce or Amplify Bias")
    
    st.markdown("""
    In this section, you've explored how TF-IDF scoring works and compared different types of terms. You've discovered:
    
    - How TF-IDF calculates a score based on term frequency in a document and inverse document frequency across the corpus
    - How rare terms receive higher IDF values, potentially overweighting them regardless of their actual importance
    - How cultural and specialized terms typically have higher IDF values than general terms
    - The significant statistical differences between cultural/specialized terms and general/common terms in our corpus
    
    Through these experiments, you've identified several ways TF-IDF can introduce or amplify bias:
    
    1. **Statistical bias from corpus composition**
       - Cultural terms are naturally less frequent in our corpus, giving them higher IDF scores
       - This might seem like an advantage, but can be misleading since the rarity is due to corpus bias, not actual information value
       - Example: "diaspora" might get a high IDF not because it's more informative, but because of under-representation
    
    2. **Contextual meaning loss**
       - TF-IDF treats words as independent units, losing important contexts
       - Cultural concepts often rely heavily on context for proper interpretation
       - Example: "passing" has specific cultural meanings in certain communities that are lost when treated as an isolated term
    
    3. **Compound term fragmentation**
       - Cultural concepts often expressed in multi-word terms (e.g., "code-switching", "critical race theory")
       - Preprocessing may split these into separate terms, affecting their TF-IDF scores
    
    4. **Term frequency thresholds**
       - Many systems ignore terms below certain frequency thresholds to optimize performance
       - This disproportionately affects culturally specific or specialized terms
    
    ### Mitigation Strategies
    
    Some approaches to address these biases include:
    
    1. **Corpus balancing**: Ensure diverse representation in the document collection
    2. **Specialized term weighting**: Adjust weights for recognized cultural or specialized terms
    3. **Phrase preservation**: Maintain multi-word expressions as single units in the index
    4. **Context-aware weighting**: Consider local context when calculating term importance
    """)

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("← Back to 'Preprocessing'", 
                  on_click=navigate_to, 
                  args=("Text Preprocessing Bias",),
                  key="tfidf_prev")
    with col3:
        st.button("Continue to 'System Bias' →", 
                  on_click=navigate_to, 
                  args=("Putting It All Together",),
                  key="tfidf_next")

def show_system_bias():
    st.title("Putting It All Together: System-Level Bias")
    
    st.markdown("""
    We've examined individual components of an Information Retrieval system and identified potential biases in each:
    
    1. **Selection Bias**: Our corpus is skewed toward older, Western-centric texts
    2. **Preprocessing Bias**: Cultural terms, names, and non-English words lose distinctive features
    3. **Statistical Bias**: TF-IDF can amplify underrepresentation of cultural terms
    
    Now, let's see how these biases compound through the system to affect final search results.
    """)
    
    # Toggle between different views
    visualization_type = st.radio(
        "Choose visualization:",
        ["Bias Points Focus", "Simplified IR System", "Detailed IR System"],
        horizontal=True,
        index=0  # Default to bias points view for this section
    )
    
    if visualization_type == "Bias Points Focus":
        display_ir_system_visualization("bias_points")
    elif visualization_type == "Simplified IR System":
        display_ir_system_visualization("simplified")
    else:
        display_ir_system_visualization("detailed")
    
    # Explanation of bias compounding
    st.subheader("How Bias Compounds Through the System")
    
    st.markdown("""
    Bias doesn't just appear at individual points in the system; it compounds and amplifies:
    
    1. **Corpus bias** → The foundation of all search results is skewed
    2. **Preprocessing bias** → Further distorts representation of cultural terms
    3. **Statistical bias** → Amplifies distortions through term weights
    4. **Ranking bias** → Final results may systematically disadvantage certain perspectives
    
    This creates a cascade effect where bias at each stage can multiply the effects of bias at other stages.
    """)
    
    # Import the required functions from system_bias.py
    from utils.system_bias import (
        run_bias_comparison_search, 
        trace_term_through_system, 
        run_mitigated_search, 
        run_system_bias_analysis
    )
    
    # Set up tabs for different query examples
    query_tabs = st.tabs(["Custom Search", "Cultural Terms Example", "Names Example", "Mixed Language Example"])
    
    with query_tabs[0]:
        # Custom search query input
        query = st.text_input("Enter a search query:")
        
        # Execute search if query is provided
        if query:
            run_bias_comparison_search(query, search_index)
    
    with query_tabs[1]:
        st.markdown("""
        This example demonstrates how cultural terms are processed differently by the search system.
        Try comparing results for terms like "diaspora", "indigenous", or "afrofuturism".
        """)
        
        cultural_query = st.selectbox(
            "Select a cultural term to search for:",
            ["diaspora", "indigenous", "afrofuturism", "hip-hop", "latinx"]
        )
        
        if st.button("Run Cultural Terms Example"):
            run_bias_comparison_search(cultural_query, search_index)
    
    with query_tabs[2]:
        st.markdown("""
        This example demonstrates how names from different cultures are processed differently.
        Try comparing results for different naming patterns.
        """)
        
        name_query = st.selectbox(
            "Select a name pattern to search for:",
            ["Maria Rodriguez-Lopez", "O'Connor", "Nguyễn", "DeAndre"]
        )
        
        if st.button("Run Names Example"):
            run_bias_comparison_search(name_query, search_index)
    
    with query_tabs[3]:
        st.markdown("""
        This example demonstrates how mixed language queries are affected by bias.
        Try comparing results for terms with non-English elements.
        """)
        
        mixed_query = st.selectbox(
            "Select a mixed language query:",
            ["café culture", "déjà vu", "niño education", "São Paulo"]
        )
        
        if st.button("Run Mixed Language Example"):
            run_bias_comparison_search(mixed_query, search_index)

    # Detailed bias example with step-by-step visualization
    st.subheader("Step-by-Step Bias Trace Example")
    
    st.markdown("""
    Let's trace how bias affects a specific search term as it moves through the IR pipeline.
    This example will show each transformation and how it impacts retrievability.
    """)
    
    example_term = st.selectbox(
        "Select a term to trace through the system:",
        ["African-American", "Indigenous", "Latinx culture", "María's story"]
    )
    
    if st.button("Trace Term Through System"):
        trace_term_through_system(example_term, search_index)
    
    # Bias mitigation strategies
    st.subheader("Exploring Bias Mitigation Strategies")
    
    st.markdown("""
    How might we reduce bias in search systems? Let's explore some mitigation strategies
    and see their impact on search results.
    """)
    
    # Mitigation toggles
    mitigation_options = {
        "preserve_case": "Preserve case information (protects proper nouns)",
        "custom_stemming": "Use custom stemming for cultural terms",
        "context_aware": "Enable context-aware indexing",
        "term_boosting": "Apply boosting for underrepresented terms",
        "diverse_corpus": "Use more diverse corpus weighting"
    }
    
    selected_mitigations = st.multiselect(
        "Select bias mitigation strategies to apply:",
        list(mitigation_options.keys()),
        format_func=lambda x: mitigation_options[x]
    )
    
    mitigation_query = st.text_input("Enter a query to test with mitigation strategies:", "indigenous peoples")
    
    if st.button("Run Mitigated Search"):
        run_mitigated_search(mitigation_query, selected_mitigations, search_index)
    
    # System-level bias analysis tool
    st.subheader("System-Level Bias Analysis Tool")
    
    st.markdown("""
    This tool allows you to analyze the overall bias in search results for different types of queries.
    Choose a category of terms to analyze and see how the system treats them differently.
    """)
    
    bias_analysis_options = {
        "cultural_vs_general": "Cultural terms vs. General terms",
        "western_vs_nonwestern": "Western vs. Non-Western concepts",
        "english_vs_noneng": "English vs. Non-English terms",
        "gender_analysis": "Gender representation analysis"
    }
    
    selected_analysis = st.selectbox(
        "Select an analysis type:",
        list(bias_analysis_options.keys()),
        format_func=lambda x: bias_analysis_options[x]
    )
    
    if st.button("Run System Bias Analysis"):
        run_system_bias_analysis(selected_analysis, search_index)

    # Hypothesis box
    st.subheader("Form a Hypothesis")
    st.markdown("""
    **How might these biases interact to affect different types of search queries?**

    Consider the following questions as you explore the examples:
    - How might a search for cultural concepts be affected differently than general concepts?
    - What happens when multiple bias factors affect the same query (e.g., a non-English name)?
    - Which types of queries do you think would be most disadvantaged by these compounding biases?

    Use the interactive examples below to test your hypothesis.
    """)

    hypothesis = st.text_area("Your hypothesis about system-level bias:", height=150)
    
    # Educational summary 
    st.header("What We've Learned: Understanding System-Level Bias")
    
    st.markdown("""
    In this section, you've witnessed how bias compounds through the entire IR system pipeline. Your exploration has included:
    
    - Comparing standard search results with bias-mitigated alternatives
    - Tracing specific terms through each stage of the IR pipeline to observe transformations
    - Testing various mitigation strategies and their impact on search results
    - Analyzing system-level bias across different categories of terms
    
    These experiments have demonstrated how biases interact throughout the system:
    
    ### System-Level Bias in Information Retrieval
    
    System-level bias occurs when multiple components of a system contain biases that interact 
    and reinforce each other. In information retrieval, this can lead to:
    
    1. **Compounding effects**: Minor biases at each step multiply to create major disparities in results
    
    2. **Self-reinforcing patterns**: Popular, biased results become more popular, creating feedback loops
    
    3. **Invisible disadvantages**: Certain types of queries are systematically disadvantaged in ways 
    that are difficult to detect without careful analysis
    
    4. **Emergent bias**: New biases that weren't present in any individual component can emerge from interactions
    
    ### Detecting System-Level Bias
    
    Techniques for identifying system-level bias include:
    
    - Fairness audits with diverse query sets
    - Compare results across different demographic groups
    - Measure representation of different perspectives in search results
    - Track performance for culturally specific versus general queries
    
    ### Mitigating System-Level Bias
    
    Addressing system-level bias requires a holistic approach:
    
    - Diverse data collection across all dimensions (temporal, cultural, linguistic)
    - Culture-aware preprocessing that preserves important distinctions
    - Context-sensitive indexing approaches
    - Fairness-aware ranking algorithms
    - Regular auditing and monitoring of system behavior
    """)

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("← Back to 'TF-IDF'", 
                  on_click=navigate_to, 
                  args=("Statistical Bias: TF-IDFs",),
                  key="system_prev")
    with col3:
        st.button("Continue to 'Final Reflection' →", 
                  on_click=navigate_to, 
                  args=("Final Reflection",),
                  key="system_next")

def show_reflection():
    st.title("Final Reflection: Building More Equitable IR Systems")
    
    st.markdown("""
    Throughout this interactive essay, we've explored how bias can enter Information Retrieval (IR) systems 
    at multiple points and compound through the pipeline. Let's reflect on what we've learned and consider 
    how we might build more equitable search systems in the future.
    """)
    
    # Summary of key learnings
    st.subheader("Key Takeaways")
    
    st.markdown("""
    We've examined three main types of bias in IR systems:
    
    1. **Selection Bias**: We saw how the composition of our corpus (Project Gutenberg texts) influences search 
    results, with Western, male, and older perspectives dominating the collection.
    
    2. **Preprocessing Bias**: We learned how standard preprocessing steps like case folding, punctuation removal, 
    and stemming can disproportionately affect cultural terms, names, and non-English words.
    
    3. **Statistical Bias**: We explored how TF-IDF calculations can inadvertently amplify existing corpus biases 
    by treating underrepresented terms as highly distinctive, regardless of their actual information value.
    
    Most importantly, we've seen how these biases compound and interact, creating a system where certain types 
    of queries and certain information needs are systematically disadvantaged.
    """)
    
    # Reflection questions
    st.subheader("Reflection Questions")
    
    questions = [
        "How might biased search results affect different user groups?",
        "What responsibility do search engine designers have to address algorithmic bias?",
        "How might we balance technical efficiency with fairness and equity in IR systems?",
        "What other technologies might be affected by similar types of algorithmic bias?",
        "How can users become more aware of potential bias in search results?"
    ]
    
    selected_question = st.selectbox("Choose a reflection question to answer:", questions)
    
    st.text_area("Your reflection:", height=150, key="reflection_response")
    
    # Strategies for more equitable IR systems
    st.subheader("Building More Equitable IR Systems")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Technical Strategies")
        st.markdown("""
        * **Diverse corpus selection**: Ensure document collections represent varied perspectives, time periods, and cultures
        
        * **Culture-aware preprocessing**: Develop methods that preserve important cultural distinctions
        
        * **Context-sensitive indexing**: Maintain relationships between terms rather than treating them as independent units
        
        * **Fairness metrics**: Develop and monitor metrics for equity and representation in search results
        
        * **Alternative ranking functions**: Explore alternatives to TF-IDF that are less sensitive to corpus biases
        
        * **User feedback incorporation**: Learn from diverse user interactions to improve relevance for all groups
        """)
    
    with col2:
        st.markdown("#### Policy & Design Strategies")
        st.markdown("""
        * **Diverse development teams**: Include people from varied backgrounds in system design and evaluation
        
        * **Transparency**: Clearly document system limitations and potential biases
        
        * **User education**: Help users understand how search works and its potential limitations
        
        * **Regular bias audits**: Conduct regular testing for potential bias across different query types
        
        * **Expanded result diversity**: Design interfaces that encourage exploration of diverse perspectives
        
        * **Community involvement**: Include communities potentially affected by bias in the design process
        """)
    
    # Future research directions
    st.subheader("Future Directions")
    
    st.markdown("""
    The field of fair and unbiased information retrieval continues to evolve. Some promising research directions include:
    
    * **Neural IR methods** that better capture semantic relationships while mitigating bias
    
    * **Personalization** approaches that consider individual needs without reinforcing systemic biases
    
    * **Explainable IR** techniques that help users understand why certain results appear
     
    * **Cross-cultural IR** systems designed from the ground up to serve diverse global users
    
    * **User control** interfaces that allow users to adjust system behavior based on their needs
    """)
    
    # Resources for further learning
    st.subheader("Resources for Further Learning")
    
    st.markdown("""
    If you'd like to explore these topics further, here are some excellent resources:
    
    * **Books**:
      * "Algorithms of Oppression" by Safiya Umoja Noble
      * "Race After Technology" by Ruha Benjamin
      * "Data Feminism" by Catherine D'Ignazio and Lauren F. Klein
    
    * **Academic Papers that I Found Particularly Helpful**:
      * Friedman & Nissenbaum, "Bias in Computer Systems" (1996)
      * Mehrabi et al., "A Survey on Bias and Fairness in Machine Learning" (2021)
      * Hutchinson et al., "Towards Accountability for Machine Learning Datasets" (2021)
    
    * **Online Resources**:
      * [Algorithmic Justice League](https://www.ajl.org/)
      * [Data & Society](https://datasociety.net/)
      * [FAT* Conference](https://facctconference.org/) (Fairness, Accountability, and Transparency)
    """)
    
    # Final thoughts
    st.markdown("""
    ### Final Thoughts
    
    The algorithms that power search engines are neither neutral nor objective - they reflect the data 
    they're built on and the choices made by their designers. By understanding the sources of bias in 
    information retrieval systems, we can work toward creating more equitable technology that serves 
    diverse users fairly.
    
    As users, creators, and critics of these systems, we all have a role to play in demanding and 
    building technology that works for everyone.
    """)

    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button("← Back to 'System Bias'", 
                  on_click=navigate_to, 
                  args=("Putting It All Together",),
                  key="reflection_prev")
    with col2:
        st.button("Return to Start", 
                  on_click=navigate_to, 
                  args=("Introduction",),
                  key="reflection_home")

if __name__ == "__main__":
    main()