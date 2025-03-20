# utils/ir_visualizations.py
import streamlit as st
import graphviz

def get_simplified_ir_flowchart():
    """
    Return a simplified Graphviz code for the IR system flowchart
    focusing only on the main components discussed in the essay
    
    Returns:
    --------
    str
        Simplified Graphviz DOT code
    """
    return """
    digraph IR_System_Simplified {
        rankdir=TD;
        node [shape=box, style=filled, fillcolor=white, fontname="Arial"];
        edge [fontname="Arial", fontsize=10];
        
        # Main components
        Corpus [label="Document Corpus", fillcolor="#ffcccc"];
        Preproc [label="Text Preprocessing", fillcolor="#ffcccc"];
        Index [label="Indexing", fillcolor="#ffcccc"];
        QueryProc [label="Query Processing", fillcolor="#ffcccc"];
        TFIDF [label="TF-IDF Calculation", fillcolor="#ffcccc"];
        Results [label="Search Results"];
        
        # Flow
        Corpus -> Preproc [label="Selection Bias"];
        Preproc -> Index [label="Preprocessing Bias"];
        Index -> TFIDF [label="Context Loss"];
        QueryProc -> TFIDF [label="Same Preprocessing Bias"];
        TFIDF -> Results [label="Statistical Bias"];
        
        # Legend
        subgraph cluster_legend {
            label="Legend";
            node [shape=box, style=filled];
            BiasPoint [label="Bias Entry Point", fillcolor="#ffcccc"];
        }
    }
    """

def get_bias_points_visualization():
    """
    Return Graphviz code that focuses specifically on the three 
    main bias points discussed in the essay
    
    Returns:
    --------
    str
        Graphviz DOT code for bias points visualization
    """
    return """
    digraph Bias_Points {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor=white, fontname="Arial"];
        edge [fontname="Arial", fontsize=11];
        
        # Create a custom invisible node to force layout
        invisible [style=invis, shape=point, width=0, height=0];
        
        # Three main bias points with detailed descriptions
        CorpusBias [label="1. Corpus Bias", fillcolor="#ffcccc", penwidth=2];
        PreprocessingBias [label="2. Preprocessing Bias", fillcolor="#ffcccc", penwidth=2];
        StatisticalBias [label="3. Statistical Bias", fillcolor="#ffcccc", penwidth=2];
        
        # Connect them to the invisible node to force layout
        invisible -> CorpusBias [style=invis];
        invisible -> PreprocessingBias [style=invis];
        invisible -> StatisticalBias [style=invis];
        
        # Examples for each bias type
        CorpusBias_Example [label="Example:\nHistorical, Western-centric texts\ndominate the corpus", shape=note, fillcolor="#ffffcc"];
        PreprocessingBias_Example [label="Example:\n'María Rodríguez-López' → 'maria rodriguez lopez'\nLoses diacritics, case, and hyphenation", shape=note, fillcolor="#ffffcc"];
        StatisticalBias_Example [label="Example:\nCultural terms get higher IDF weights\ndue to corpus underrepresentation", shape=note, fillcolor="#ffffcc"];
        
        # Connect bias types to examples
        CorpusBias -> CorpusBias_Example;
        PreprocessingBias -> PreprocessingBias_Example;
        StatisticalBias -> StatisticalBias_Example;
        
        # Compounding effect arrow connecting all three
        edge [color="#ff6666", penwidth=2, style=dashed];
        CorpusBias -> PreprocessingBias [label="Compounds"];
        PreprocessingBias -> StatisticalBias [label="Compounds"];
        
        # Add final result showing system-level bias
        SystemBias [label="System-Level\nBias", fillcolor="#ff9999", penwidth=3, fontsize=14];
        StatisticalBias -> SystemBias [label="Results in"];
    }
    """

def get_detailed_ir_flowchart():
    """
    Return the Graphviz code for the detailed IR system flowchart
    
    Returns:
    --------
    str
        Graphviz DOT code
    """
    return """
    digraph IR_System {
        rankdir=TD;
        node [shape=box, style=filled, fillcolor=white];
        
        subgraph cluster_input {
            label="Input and Preprocessing";
            A [label="Document Collection", fillcolor="#ffcccc"];
            B [label="Raw Text"];
            C [label="Text Preprocessing", fillcolor="#ffcccc"];
            D [label="Processed Text"];
            
            A -> B [label="Selection Bias"];
            B -> C;
            C -> D [label="Stemming Bias"];
        }
        
        subgraph cluster_indexing {
            label="Indexing System";
            E [label="Build Inverted Index", fillcolor="#ffcccc"];
            F [label="Word-Document Mapping"];
            F1 [label="Positional Index"];
            
            D -> E;
            E -> F [label="Context Loss"];
            F -> F1;
        }
        
        subgraph cluster_query {
            label="Query Processing";
            G [label="User Query"];
            H [label="Query Preprocessing", fillcolor="#ffcccc"];
            I [label="Processed Query"];
            
            G -> H;
            H -> I [label="Same Stemming Bias"];
        }
        
        subgraph cluster_retrieval {
            label="Retrieval Methods";
            J1 [label="Boolean Retrieval", fillcolor="#ffcccc"];
            J2 [label="Phrase Query Retrieval", fillcolor="#ffcccc"];
            J3 [label="TF-IDF Calculation", fillcolor="#ffcccc"];
            K1 [label="Boolean Results"];
            K2 [label="Phrase Results"];
            K3 [label="TF-IDF Vectors"];
            L [label="Cosine Similarity", fillcolor="#ffcccc"];
            K4 [label="Similarity Results"];
            
            I -> J1;
            F -> J1;
            J1 -> K1 [label="Exact Match Bias"];
            
            I -> J2;
            F1 -> J2;
            J2 -> K2 [label="Strict Order Bias"];
            
            I -> J3;
            F -> J3;
            J3 -> K3 [label="Statistical Bias"];
            K3 -> L;
            L -> K4 [label="Vector Space Bias"];
        }
        
        subgraph cluster_ranking {
            label="Result Integration";
            M [label="Result Integration"];
            N [label="Final Search Results"];
            
            K1 -> M;
            K2 -> M;
            K4 -> M;
            M -> N;
        }
        
        note [label="Nodes in red indicate\nbias entry points", shape=note, fillcolor="#fff"];
    }
    """

def display_ir_system_visualization(visualization_type="simplified"):
    """
    Display the selected IR system visualization
    
    Parameters:
    -----------
    visualization_type : str
        Type of visualization to display ('simplified', 'bias_points', or 'detailed')
    """
    import graphviz
    
    if visualization_type == "simplified":
        st.subheader("Simplified IR System Flowchart")
        graph_code = get_simplified_ir_flowchart()
        st.graphviz_chart(graph_code)
        st.caption("Simplified flowchart showing the main components and bias entry points")
        
    elif visualization_type == "bias_points":
        st.subheader("Bias Points Visualization")
        graph_code = get_bias_points_visualization()
        st.graphviz_chart(graph_code)
        st.caption("Visualization focused on how the three main types of bias compound through the system")
        
    else:  # detailed
        st.subheader("Detailed IR System Flowchart")
        graph_code = get_detailed_ir_flowchart()
        st.graphviz_chart(graph_code)
        st.caption("Detailed flowchart of an IR system with bias entry points highlighted in red")