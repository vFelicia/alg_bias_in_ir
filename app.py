# For Windows
# python -m venv venv ---> creates a new venv
# venv\Scripts\activate ---> just run this to activate

# PS C:\GitHubRepos\alg_bias_in_ir> python -m venv venv
# PS C:\GitHubRepos\alg_bias_in_ir> venv/Scripts/activate
# (venv) PS C:\GitHubRepos\alg_bias_in_ir> streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import PorterStemmer
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.preprocessing import preprocess_text
from utils.indexing import InvertedIndex
from utils.retrieval import search_documents
from utils.visualization import plot_corpus_stats
import pickle
import os
import time

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
    # Set page config
    st.set_page_config(
        page_title="Search Engines and Hidden Biases",
        page_icon="🔍",
        layout="wide"
    )
    
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
        st.sidebar.write("No metadata loaded")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Introduction", "The Corpus Matters", "Text Preprocessing", 
         "Understanding TF-IDF", "System-Level Bias", "Final Reflection"]
    )
    
    # Main content based on selected page
    if page == "Introduction":
        show_introduction()
    elif page == "The Corpus Matters":
        show_corpus_analysis()
    elif page == "Text Preprocessing":
        show_preprocessing()
    elif page == "Understanding TF-IDF":
        show_tfidf_calculator()
    elif page == "System-Level Bias":
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
                
    Our dataset for this interactive essay is a collection of sixty books sourced from Project Gutenberg. From these sixty books,
    we kept track of  the book's title, author, year published, gender of author, nationality of author, and book genre.
    """)
    
    # Display some stats about the loaded corpus
    st.info(f"Loaded {len(search_index.documents)} documents into the search index.")
    
    # Basic search interface
    st.subheader("Let's get you acquainted with our data and our search engine! Try a basic search. It should return the title(s) of a book.")
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

def show_corpus_analysis():
    st.title("The Corpus Matters: Project Gutenberg as a Case Study")
    
    # Explanation of corpus
    st.markdown("""
    The corpus - or collection of documents - is the foundation of any search system. For our exploration,
    we're using sixty randomly selected texts from Project Gutenberg, a digital library of free eBooks.
    
    Project Gutenberg primarily contains older, public domain works. Let's examine the characteristics of this collection.
    """)
    
    # Visualization of corpus statistics
    st.subheader("Corpus Statistics")
    
    # This would use your actual data
    tab1, tab2, tab3 = st.tabs(["Publication Dates", "Author Demographics", "Geographic Distribution"])
    
    with tab1:
        # Plot publication date distribution
        fig, ax = plt.subplots()
        # Your plotting code here
        st.pyplot(fig)
    
    with tab2:
        # Author demographics
        fig, ax = plt.subplots()
        # Your plotting code here
        st.pyplot(fig)
        
    with tab3:
        # Geographic distribution
        fig, ax = plt.subplots()
        # Your plotting code here
        st.pyplot(fig)
    
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

def show_preprocessing():
    st.title("Text Preprocessing: Where Bias Begins")
    
    st.markdown("""
    Before texts can be searched, they undergo preprocessing - a series of transformations that prepare them for efficient indexing.
    While these steps are technical necessities, they can introduce bias in subtle ways.
    
    Let's explore common preprocessing steps and see how they might affect different types of text.
    """)
    
    # Interactive preprocessing simulator
    st.subheader("Text Preprocessing Simulator")
    
    input_text = st.text_area("Enter some text to preprocess:", 
                             value="The quick brown fox jumps over the lazy dog. Isn't natural language processing AMAZING?")
    
    if input_text:
        # Step by step preprocessing visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Step 1: Lowercase Conversion**")
            lowercase_text = input_text.lower()
            st.text(lowercase_text)
        
        with col2:
            st.markdown("**Step 2: Punctuation Removal**")
            no_punct_text = re.sub(r'[^\w\s]', '', lowercase_text)
            st.text(no_punct_text)
        
        with col3:
            st.markdown("**Step 3: Stemming**")
            ps = PorterStemmer()
            tokens = nltk.word_tokenize(no_punct_text)
            stemmed_tokens = [ps.stem(token) for token in tokens]
            stemmed_text = ' '.join(stemmed_tokens)
            st.text(stemmed_text)
    
    # Case study
    st.subheader("Case Study: Impact on Cultural Terms")
    
    st.markdown("""
    Let's examine how preprocessing affects different types of terms:
    
    1. Standard English: "running", "jumped", "cars"
    2. Cultural terms: "hip-hop", "afrofuturism", "latinx"
    3. Names: "O'Connor", "Nguyen", "López-Álvarez"
    """)
    
    # Comparison table showing original and processed forms
    # Implementation here
    
    # Reflection box
    st.subheader("Reflection")
    st.markdown("""
    **What information is lost in preprocessing?**
    
    Based on the examples above, what types of terms might be disadvantaged by standard preprocessing steps?
    How might this affect search results for different user groups?
    """)
    
    reflection = st.text_area("Your reflections on preprocessing bias:", height=150)

def show_tfidf_calculator():
    st.title("Understanding TF-IDF: Numbers That Shape Results")
    
    st.markdown("""
    Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic used to reflect how important a word is to a document in a collection.
    It's one of the fundamental algorithms used in search engines to rank results.
    
    Let's explore how TF-IDF works and how it might introduce bias.
    """)
    
    # TF-IDF Calculator
    st.subheader("TF-IDF Calculator")
    
    # Word input
    word = st.text_input("Enter a word:")
    
    # Document selection
    options = ["Document 1: Pride and Prejudice excerpt", 
               "Document 2: Moby Dick excerpt",
               "Document 3: The Art of War excerpt"]
    selected_docs = st.multiselect("Select documents to compare:", options)
    
    if word and selected_docs:
        # Calculate and display TF-IDF
        st.write(f"TF-IDF scores for '{word}':")
        
        for doc in selected_docs:
            # This would use your actual implementation
            score = 0.75  # Placeholder
            st.write(f"{doc}: {score:.4f}")
        
        # Visualization
        st.bar_chart({doc: 0.75 for doc in selected_docs})  # Placeholder
    
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

def show_system_bias():
    st.title("Putting It All Together: System-Level Bias")
    
    st.markdown("""
    We've examined individual components of an Information Retrieval system and identified potential biases in each.
    Now, let's see how these biases compound through the system to affect final search results.
    
    The flowchart below illustrates the complete IR pipeline, with bias entry points highlighted in red.
    """)
    
    # Display flowchart image
    # st.image("assets/ir_flowchart.png")
    
    # Complete search system demo
    st.subheader("Complete Search System")
    
    query = st.text_input("Enter a search query to see bias effects:")
    
    bias_toggles = st.multiselect(
        "Toggle bias mitigation strategies:",
        ["Use diverse corpus", "Preserve case information", "Custom stemming for names", 
         "Context-aware indexing", "Cultural relevance boosting"]
    )
    
    if query:
        # Show search results with and without bias mitigation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Standard Results**")
            # Display standard results
            
        with col2:
            st.markdown("**Results with Bias Mitigation**")
            # Display mitigated results

def show_reflection():
    st.title("Final Reflection: How could we design more equitable IR systems?")
    
    st.markdown("""
    Throughout this interactive essay, we've explored how bias can enter information retrieval systems at various stages:
    
    1. Through the corpus selection
    2. During text preprocessing
    3. In indexing and context representation
    4. Through statistical methods like TF-IDF
    
    Now, let's think about how we might design more equitable IR systems.
    """)
    
    # Discussion questions
    st.subheader("Discussion Questions")
    
    questions = [
        "How could corpus selection be improved to reduce bias?",
        "What modifications to preprocessing could better preserve cultural information?",
        "How might ranking algorithms be adapted to account for representation issues?",
        "What role should human oversight play in search algorithms?",
        "How can we evaluate IR systems for bias?"
    ]
    
    for i, question in enumerate(questions, 1):
        st.markdown(f"**{i}. {question}**")
        st.text_area(f"Your thoughts on question {i}:", height=100, key=f"q{i}")
    
    # Resources
    st.subheader("Further Resources")
    
    st.markdown("""
    - Noble, S. U. (2018). *Algorithms of Oppression: How Search Engines Reinforce Racism*. NYU Press.
    - Friedman, B., & Nissenbaum, H. (1996). Bias in computer systems. *ACM Transactions on Information Systems*.
    - Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*.
    - [MIT Technology Review: How to Make Algorithms Fair](https://www.technologyreview.com/)
    - [Fairness in Machine Learning](https://fairmlbook.org/)
    """)

if __name__ == "__main__":
    main()