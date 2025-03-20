# Algorithmic Bias in Search Engines: Interactive Essay

This interactive essay explores how algorithmic bias manifests in information retrieval (IR) systems. Through hands-on visualizations and experiments, users can investigate how bias enters search systems at multiple stages and compounds to affect results.

VIDEO DEMO: https://youtu.be/y94UcSnDZ00

## Running the Application

In your command line, run the following steps:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

## Project Overview

This interactive essay guides users through different aspects of algorithmic bias in search engines:

1. **Introduction**: Overview of information retrieval systems and bias entry points
2. **Selection Bias and the Corpus**: How the composition of document collections affects search results
3. **Text Preprocessing Bias**: How text transformations can disproportionately affect cultural terms and names
4. **Statistical Bias: TF-IDFs**: How term weighting can amplify existing corpus biases
5. **System-Level Bias**: How biases compound through the entire system
6. **Final Reflection**: Strategies for building more equitable IR systems

## Technical Components

The application includes several interactive tools:

- **Search Engine Interface**: Test queries against a corpus of Project Gutenberg texts
- **Text Preprocessing Simulator**: See how different preprocessing steps affect various types of text
- **TF-IDF Calculator**: Explore how term weighting works and affects different types of terms
- **Bias Trace Visualization**: Follow terms through the IR pipeline to observe transformations
- **Bias Mitigation Lab**: Test how different mitigation strategies affect search results

## Data

This project uses a sample of sixty texts from Project Gutenberg, with metadata tracking:
- Book title
- Author
- Publication year
- Author gender
- Author nationality
- Book genre

## File Structure

- `app.py`: Main Streamlit application
- `utils/`: Utility modules
  - `preprocessing.py`: Text preprocessing utilities
  - `indexing.py`: Inverted index implementation
  - `tfidf.py`: TF-IDF calculation utilities
  - `retrieval.py`: Search functionality
  - `visualization.py`: Data visualization utilities
  - `ir_visualizations.py`: IR system visualizations
  - `system_bias.py`: System-level bias analysis tools
- `data/`: Contains the corpus texts and metadata
- `requirements.txt`: Required Python packages

## Requirements

This application requires:
- Python 3.7+
- Streamlit
- NLTK
- pandas
- NumPy
- Plotly
- scikit-learn
- Matplotlib

A complete list of dependencies is available in `requirements.txt`.

## Related Work

This project draws inspiration from:
- Safiya Noble's "Algorithms of Oppression" (2018)
- Friedman & Nissenbaum's "Bias in Computer Systems" (1996)

## Authorship

Vryan Feliciano -- vgfelica@stanford.edu

## Acknowledgments

This project was developed as part of EDUC 432 at Stanford University, Winter 2025, under the guidance of Dr. Hariharan Subramonyam.
