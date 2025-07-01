import nest_asyncio
nest_asyncio.apply()  # Fix for asyncio event loop error

import streamlit as st
import os
import tempfile
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_neo4j import Neo4jGraph
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="SimplyReview", layout="wide")

# Initialize session state variables
if "documents" not in st.session_state:
    st.session_state.documents = []
if "graph" not in st.session_state:
    st.session_state.graph = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "graph_client" not in st.session_state:
    st.session_state.graph_client = None
if "init" not in st.session_state:
    st.session_state.init = True

# Helper function to test Neo4j connection
def connect_to_neo4j(uri, username, password):
    try:
        # Test the connection
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
    except Exception as e:
        st.error(f"Neo4j connection failed: {str(e)}")
        return False

# Helper function to create Neo4j constraints based on version
def create_constraints(graph_client):
    try:
        # Check Neo4j version
        try:
            version_query = "CALL dbms.components() YIELD versions RETURN versions[0] as version"
            result = graph_client.query(version_query)
            version = result[0]["version"] if result else "4.0.0"  # Default to 4.0.0
        except Exception:
            # If version check fails, try alternative (for Neo4j 5.x+)
            try:
                version_query = "CALL dbms.version() YIELD version RETURN version"
                result = graph_client.query(version_query)
                version = result[0]["version"] if result else "4.0.0"
            except Exception:
                version = "4.0.0"  # Default if all else fails
        
        # Neo4j 4.0+ syntax
        if version.startswith("4") or version.startswith("5"):
            try:
                graph_client.query("""
                CREATE CONSTRAINT document_id_constraint IF NOT EXISTS
                FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
            except Exception:
                # Try alternate syntax for Neo4j 4.4+
                graph_client.query("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
            
            try:
                graph_client.query("""
                CREATE CONSTRAINT entity_name_constraint IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """)
            except Exception:
                graph_client.query("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE
                """)
            
            try:
                graph_client.query("""
                CREATE CONSTRAINT concept_name_constraint IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.name IS UNIQUE
                """)
            except Exception:
                graph_client.query("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE
                """)
        # Neo4j 3.x syntax
        else:
            graph_client.query("""
            CREATE CONSTRAINT ON (d:Document) ASSERT d.id IS UNIQUE
            """)
            
            graph_client.query("""
            CREATE CONSTRAINT ON (e:Entity) ASSERT e.name IS UNIQUE
            """)
            
            graph_client.query("""
            CREATE CONSTRAINT ON (c:Concept) ASSERT c.name IS UNIQUE
            """)
            
    except Exception as e:
            print(f"Could not create constraints: {str(e)}")

# Helper function to create knowledge graph nodes for documents
def create_document_nodes(documents, graph_client):
    if not documents or not graph_client:
        return None
    
    # Create constraints
    create_constraints(graph_client)
    
    success_count = 0
    error_count = 0
    
    # Process each document to create document nodes
    for i, doc in enumerate(documents):
        try:
            # Create Document node
            doc_id = f"doc_{i}_{hash(doc.page_content) % 10000000}"  # Limit hash size
            doc_title = doc.metadata.get("source")
            
            # Create Document node
            graph_client.query("""
            MERGE (d:Document {id: $id})
            SET d.source = $title,
                d.text = $text
            RETURN d
            """, {"id": doc_id, "title": doc_title, "text": doc.page_content[:500]})
            
            success_count += 1
        
        except Exception as e:
            error_count += 1
            st.warning(f"Error processing document {i}: {str(e)}")
    
    st.info(f"Processed {success_count} document chunks successfully with {error_count} errors")
    return True

# Relationship building method that uses embeddings directly from LangChain
def connect_documents_with_langchain(graph_client, vector_store, embeddings, similarity_threshold=0.7, max_connections=5):
    if not graph_client or not vector_store or not embeddings:
        return None
    
    with st.spinner("Creating similarity connections using LangChain embeddings..."):
        try:
            # Get all documents
            docs_query = """
            MATCH (d:Document)
            RETURN d.id as id, d.title as title, d.text as text
            """
            
            docs_result = graph_client.query(docs_query)
            
            if not docs_result:
                st.warning("No documents found")
                return False
            
            doc_count = len(docs_result)
            st.info(f"Found {doc_count} documents to process")
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process in smaller batches
            batch_size = 20
            relationship_count = 0
            
            # Get embeddings for all documents
            doc_texts = [doc["text"] for doc in docs_result]
            doc_ids = [doc["id"] for doc in docs_result]
            
            # Generate embeddings in batches to avoid memory issues
            all_embeddings = []
            for i in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[i:i+batch_size]
                batch_embeddings = embeddings.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Update progress
                progress = (i + len(batch_texts)) / len(doc_texts) * 0.5  # First half of progress
                progress_bar.progress(progress)
                status_text.text(f"Generating embeddings: {i + len(batch_texts)}/{len(doc_texts)}")
            
            # Calculate pairwise similarity and create relationships
            for i, (doc_id, embedding) in enumerate(zip(doc_ids, all_embeddings)):
                # Calculate similarity with all other documents
                similarities = []
                for j, other_embedding in enumerate(all_embeddings):
                    if i != j:  # Skip self
                        # Calculate cosine similarity
                        similarity = np.dot(embedding, other_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                        )
                        similarities.append((doc_ids[j], similarity))
                
                # Sort by similarity (highest first)
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Take top matches that meet threshold
                top_matches = []
                for other_id, similarity in similarities[:max_connections]:
                    if similarity >= similarity_threshold:
                        top_matches.append((other_id, similarity))
                
                # Create relationships in Neo4j
                for other_id, similarity in top_matches:
                    try:
                        relation_query = """
                        MATCH (d1:Document {id: $id1}), (d2:Document {id: $id2})
                        MERGE (d1)-[r:SIMILAR_TO]->(d2)
                        SET r.score = $score
                        """
                        
                        graph_client.query(relation_query, {
                            "id1": doc_id,
                            "id2": other_id,
                            "score": float(similarity)
                        })
                        
                        relationship_count += 1
                    except Exception as e:
                        st.warning(f"Error creating relationship: {str(e)}")
                
                # Update progress
                progress = 0.5 + (i + 1) / len(doc_ids) * 0.5  # Second half of progress
                progress_bar.progress(progress)
                status_text.text(f"Creating relationships: {i+1}/{len(doc_ids)} documents... Created {relationship_count} relationships so far")
            
            st.success(f"Created {relationship_count} similarity relationships between documents")
            return True
            
        except Exception as e:
            st.error(f"Error connecting documents with LangChain: {str(e)}")
            return False

# Improved function to visualize the knowledge graph
def visualize_knowledge_graph(graph_client, max_nodes=100):
    if not graph_client:
        return None
    
    try:
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Get relationship counts for each document
        rel_count_query = f"""
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[r:SIMILAR_TO]-()
        WITH d, count(r) as rel_count
        RETURN d.id as id, d.source as source, rel_count
        ORDER BY rel_count DESC
        LIMIT {max_nodes}
        """
        
        # This prioritizes documents with relationships
        nodes_result = graph_client.query(rel_count_query)
        
        if not nodes_result:
            st.warning("No document nodes found in the database.")
            return None
        
        # Add all nodes first with size based on relationship count
        for row in nodes_result:
            node_id = row["id"]
            title = row["source"]
            rel_count = row["rel_count"]
            
            # Truncate long titles
            title_display = title[:20] + "..." if len(title) > 20 else title
            
            # Add node with relationship count and connected status
            G.add_node(node_id, 
                      label=title_display, 
                      title=title, 
                      rel_count=rel_count,
                      connected=(rel_count > 0))
        
        # Now get and add relationships between these nodes
        node_ids = [row["id"] for row in nodes_result]
        
        # Create a parameterized query with multiple OR conditions
        ids_param_list = ", ".join([f"'{node_id}'" for node_id in node_ids])
        
        rel_query = f"""
        MATCH (d1:Document)-[r:SIMILAR_TO]->(d2:Document)
        WHERE d1.id IN [{ids_param_list}] AND d2.id IN [{ids_param_list}]
        RETURN d1.id as source_id, d2.id as target_id, r.score as similarity
        """
        
        rel_result = graph_client.query(rel_query)
        
        has_relationships = len(rel_result) > 0
        
        # Add edges with weight based on similarity
        for row in rel_result:
            source_id = row["source_id"]
            target_id = row["target_id"]
            similarity = row["similarity"]
            
            # Add edge with weight based on similarity (higher similarity = stronger connection)
            if source_id in G and target_id in G:  # Ensure both nodes exist
                G.add_edge(source_id, target_id, weight=similarity, width=similarity*5)
        
        # Count connected nodes and isolated nodes
        connected_nodes = [n for n, attr in G.nodes(data=True) if attr.get('connected', False)]
        isolated_nodes = [n for n, attr in G.nodes(data=True) if not attr.get('connected', False)]
        
        # If no relationships found, return early
        if not has_relationships:
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, node_size=200, node_color='lightgray', alpha=0.8)
            labels = {n: G.nodes[n]['label'] for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
            plt.title("Document Knowledge Graph (No Relationships Found)")
            plt.axis('off')
            return plt
        
        # Calculate node sizes based on relationship count for better visualization
        node_sizes = {}
        for node in G:
            rel_count = G.nodes[node].get('rel_count', 0)
            # Base size + additional size based on relationship count
            node_sizes[node] = 200 + min(rel_count * 50, 300)  # Cap at a reasonable size
        
        # Create visualization with improved layout
        plt.figure(figsize=(12, 8))
        
        # Use force-directed layout with customized parameters
        pos = nx.spring_layout(G, k=0.2, iterations=100, seed=42)
        
        # Draw connected nodes with size based on relationships
        if connected_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=connected_nodes,
                                 node_size=[node_sizes[n] for n in connected_nodes], 
                                 node_color='cornflowerblue', 
                                 alpha=0.8)
        
        # Draw isolated nodes in gray and smaller
        if isolated_nodes:
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=isolated_nodes,
                                 node_size=150, 
                                 node_color='lightgray', 
                                 alpha=0.6)
        
        # Draw edges with width and color based on similarity
        if has_relationships:
            edges = G.edges(data=True)
            
            # Create edge color map based on similarity
            edge_colors = [data['weight'] for _, _, data in edges]
            
            # Calculate edge widths - thicker for higher similarity
            edge_widths = [data['weight'] * 3 for _, _, data in edges]
            
            nx.draw_networkx_edges(G, pos, 
                                 width=edge_widths, 
                                 alpha=0.7, 
                                 edge_cmap=plt.cm.Blues,
                                 edge_color=edge_colors,
                                 arrows=True, 
                                 arrowsize=15,
                                 connectionstyle='arc3,rad=0.1')  # Curved edges for better visibility
        
        # Draw labels with contrasting backgrounds for readability
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        
        # Add white background to labels for better visibility
        text_items = []
        for node, label in labels.items():
            text_items.append(plt.text(pos[node][0], pos[node][1], label,
                             fontsize=8, 
                             ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", 
                                      fc='white', 
                                      ec="none", 
                                      alpha=0.7)))
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cornflowerblue', markersize=10, label='Connected Documents'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='Isolated Documents'),
            plt.Line2D([0], [0], color='blue', lw=2, alpha=0.7, label='Similarity Connection')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title("Document Knowledge Graph", fontsize=16)
        plt.axis('off')
        
        # Add connection statistics
        stats_text = f"Connected Documents: {len(connected_nodes)} | Isolated Documents: {len(isolated_nodes)}"
        if has_relationships:
            stats_text += f" | Total Connections: {len(rel_result)}"
            
        plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        return plt
        
    except Exception as e:
        st.error(f"Error visualizing knowledge graph: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
    
# Delete the entire knowledge graph
def delete_knowledge_graph(graph_client):
    if not graph_client:
        return False
    
    with st.spinner("Deleting entire knowledge graph..."):
        try:
            # Delete all Document nodes and their relationships
            delete_query = """
            MATCH (d:Document)
            DETACH DELETE d
            """
            
            graph_client.query(delete_query)
            
            # Delete any Entity nodes and their relationships
            delete_entities_query = """
            MATCH (e:Entity)
            DETACH DELETE e
            """
            
            graph_client.query(delete_entities_query)
            
            # Delete any Concept nodes and their relationships
            delete_concepts_query = """
            MATCH (c:Concept)
            DETACH DELETE c
            """
            
            graph_client.query(delete_concepts_query)
            
            # Reset session state
            st.session_state.documents = []
            st.session_state.graph = None
            
            st.success("Knowledge graph deleted successfully")
            return True
            
        except Exception as e:
            st.error(f"Error deleting knowledge graph: {str(e)}")
            return False

# Custom hybrid search function
def hybrid_search(vector_store, query, k=5, alpha=0.5):
    """
    Custom implementation of hybrid search that combines vector and keyword search results.
    
    Args:
        vector_store: The Neo4jVector instance
        query: The search query
        k: Number of results to return
        alpha: Balance between vector (alpha) and keyword (1-alpha) search
            0.0 = pure keyword, 1.0 = pure vector
    
    Returns:
        Combined and re-ranked list of search results
    """
    # Get twice as many results from each method to ensure we have enough candidates
    k_extended = min(k * 2, 20) 
    
    # Get vector search results
    vector_results = vector_store.similarity_search_with_score(query, k=k_extended)
    
    # Get keyword search results
    try:
        keyword_results = vector_store.keyword_search(query, k=k_extended)
        # Add dummy scores if not present
        keyword_results_with_score = []
        for doc in keyword_results:
            if 'score' not in doc.metadata:
                # Create a copy with a default score
                doc_copy = doc.copy()
                doc_copy.metadata['score'] = 0.5  # Default score
                keyword_results_with_score.append((doc_copy, doc_copy.metadata['score']))
            else:
                keyword_results_with_score.append((doc, doc.metadata['score']))
    except Exception:
        # If keyword search fails, just use vector results
        return [doc for doc, _ in vector_results[:k]]
    
    # Combine results
    all_docs = {}
    
    # Process vector results
    for doc, score in vector_results:
        doc_id = hash(doc.page_content)
        all_docs[doc_id] = {
            'doc': doc,
            'vector_score': score,
            'keyword_score': 0.0,
            'combined_score': 0.0
        }
    
    # Process keyword results
    for doc, score in keyword_results_with_score:
        doc_id = hash(doc.page_content)
        if doc_id in all_docs:
            all_docs[doc_id]['keyword_score'] = score
        else:
            all_docs[doc_id] = {
                'doc': doc,
                'vector_score': 0.0,
                'keyword_score': score,
                'combined_score': 0.0
            }
    
    # Calculate combined scores
    for doc_id, doc_data in all_docs.items():
        # Normalize vector scores if needed (assuming higher is better)
        if doc_data['vector_score'] > 1.0:
            vector_score_norm = 1.0 / max(1.0, doc_data['vector_score'])  # Invert if distance
        else:
            vector_score_norm = doc_data['vector_score']  # Already normalized
            
        # Calculate weighted combined score
        doc_data['combined_score'] = (
            alpha * vector_score_norm + 
            (1 - alpha) * doc_data['keyword_score']
        )
        
        # Update the document's metadata with the scores
        doc_data['doc'].metadata['vector_score'] = doc_data['vector_score']
        doc_data['doc'].metadata['keyword_score'] = doc_data['keyword_score']
        doc_data['doc'].metadata['score'] = doc_data['combined_score']
    
    # Sort by combined score and return top k
    sorted_results = sorted(
        [doc_data['doc'] for doc_data in all_docs.values()],
        key=lambda x: x.metadata['score'],
        reverse=True  # Higher score is better
    )
    
    return sorted_results[:k]

def main_runner():
    # Sidebar configuration
    with st.sidebar:
        st.title("RAG Configuration")
        # Neo4j Connection Settings
        neo4j_uri = os.getenv("NEO4J_URL")
        neo4j_username = os.getenv("NEO4J_USERNAME")
        neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # LLM Settings
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        model_name = 'mistral-saba-24b'

        # Create connection button
        with st.spinner('Setting up neo4j and LLM'):
            if st.session_state.init and (neo4j_uri and neo4j_username and neo4j_password and groq_api_key):
                print('\nINIT\n')
                if not groq_api_key:
                    st.error("Please provide a Groq API key")
                    return
                    
                try:
                    # Test Neo4j connection first
                    if not connect_to_neo4j(neo4j_uri, neo4j_username, neo4j_password):
                        return
                        
                    # Initialize embeddings
                    st.session_state.embeddings = HuggingFaceEmbeddings()
                    
                    # Set up Neo4j vector store
                    try:
                        st.session_state.vector_store = Neo4jVector.from_existing_graph(
                            embedding=st.session_state.embeddings,
                            url=neo4j_uri,
                            username=neo4j_username,
                            password=neo4j_password,
                            index_name="document_embedding",
                            node_label="Document",
                            text_node_properties=["text"],
                            embedding_node_property="embedding",
                        )
                    except Exception as e:
                        # If vector store doesn't exist, create it
                        st.warning(f"Vector store not found, will be created when documents are added: {str(e)}")
                        st.session_state.vector_store = Neo4jVector.from_texts(
                            texts=["Initialize vector store"],
                            embedding=st.session_state.embeddings,
                            url=neo4j_uri,
                            username=neo4j_username,
                            password=neo4j_password,
                            index_name="document_embedding",
                            node_label="Document",
                            text_node_property="text",
                            embedding_node_property="embedding",
                        )
                    
                    # Initialize Neo4j graph client (for document nodes only)
                    st.session_state.graph_client = Neo4jGraph(
                        url=neo4j_uri,
                        username=neo4j_username,
                        password=neo4j_password,
                    )
                    
                    # Test Groq API key
                    try:
                        # Initialize ChatGroq
                        llm = ChatGroq(
                            model_name=model_name,
                            temperature=0,
                            api_key=groq_api_key,
                        )

                        # Test with a simple query
                        test_response = llm.invoke([HumanMessage(content="Say hello")])
                    except Exception as e:
                        st.error(f"Groq API connection failed: {str(e)}")
                        return
                    
                    st.success("Successfully connected to Neo4j and initialized the LLM!")
                    st.session_state.init = False
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
            
        # Model Selection
        model_name = st.selectbox(
            "Select Model",
            ["mistral-saba-24b", "llama-3.3-70b-versatile", "other"],
            help="Choose the model to use for LLM queries"
        )
        if model_name == 'other':
            model_name = st.text_input("Model Name")

        # Document processing settings
        with st.expander('Document Processing'):
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=2000)
            chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=50)

        with st.expander('KG Building'):
            similarity_threshold = st.slider("Similarity Threshold", 0.5, 0.95, 0.7, 0.05,
                                        help="Minimum cosine similarity score to create a connection")
            max_connections = st.slider("Max Connections per Document", 1, 20, 5,
                                    help="Maximum number of connections per document")
            
            if st.button("Create Document Similarity Connections"):
                if st.session_state.graph_client is None or st.session_state.vector_store is None:
                    st.error("Please connect to Neo4j and process documents first!")
                else:
                    # Call the appropriate function based on user selection
                    if st.session_state.embeddings is None:
                        st.error("Embeddings not initialized. Please reconnect to Neo4j.")
                    else:
                        result = connect_documents_with_langchain(
                            st.session_state.graph_client,
                            st.session_state.vector_store,
                            st.session_state.embeddings,
                            similarity_threshold,
                            max_connections
                        )
                    
                    if result:
                        st.success("Document similarity connections created successfully!")
            
        with st.expander('KG Visualisation'):            
            max_nodes = st.slider("Maximum Nodes to Display", 10, 500, 100,
                                help="Larger graphs may take longer to render")
            
        with st.expander('Search Configuration'):
            # Add type of search
            alpha = st.slider(
                "Semantic vs Keyword Balance",
                0.0, 1.0, 0.5, 0.1,
                help="0 = Pure Keyword, 1 = Pure Semantic"
            )

            # Add number of documents to be retrieved
            top_k = st.slider(
                "Number of Results", 
                1, 20, 5,
                help="Maximum number of documents to retrieve"
            )
            
            # Add context size option
            context_size = st.slider(
                "Context Size",
                100, 1000, 500,
                100,
                help="Maximum characters to extract around matched text"
            )
            
            # LLM answer settings
            generate_answer = st.checkbox("Generate AI Answer", value=True)
            if generate_answer:
                temperature = st.slider(
                    "Answer Creativity", 
                    0.0, 1.0, 0.0, 0.1,
                    help="Higher values make output more creative"
                )
                
        with st.expander('Database Maintenance'):
            # Make delete operation require confirmation
            delete_confirmed = st.checkbox("I understand this will delete ALL document nodes and relationships")
            
            if st.button("Delete Entire Knowledge Graph", type="primary", disabled=not delete_confirmed):
                if st.session_state.graph_client is None:
                    st.error("Please connect to Neo4j first!")
                elif delete_confirmed:
                    # Delete the entire graph
                    delete_knowledge_graph(st.session_state.graph_client)
                else:
                    st.warning("Please confirm deletion by checking the box")

    # Main app
    st.title("SimlpyReview : Literature Review made easy!")

    tab1, tab2, tab3 = st.tabs(["Upload & Process", "Document Store", "Query"])

    # Tab 1: Document Upload and Processing
    with tab1:
        st.header("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            if st.session_state.vector_store is None:
                st.error("Please configure and connect to Neo4j first!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                processed_count = 0
                error_count = 0
                
                # Process each document
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Create a temporary file
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                            tmp_file.write(file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Load document based on file type
                        try:
                            if file.name.endswith('.pdf'):
                                loader = PyPDFLoader(tmp_path)
                            elif file.name.endswith('.docx'):
                                loader = Docx2txtLoader(tmp_path)
                            elif file.name.endswith('.txt'):
                                loader = TextLoader(tmp_path)
                            else:
                                continue
                            
                            documents = loader.load()

                            # Split documents
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            docs = text_splitter.split_documents(documents)
                            
                            # Add metadata to track document source
                            for doc in docs:
                                doc.metadata["source"] = file.name
                            
                            # Add to Neo4j Vector Store
                            st.session_state.vector_store.add_documents(docs)
                            
                            # Add to session state
                            st.session_state.documents.extend(docs)
                            
                            processed_count += 1
                            
                        except Exception as e:
                            error_count += 1
                            st.error(f"Error processing {file.name}: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text(f"Processing complete! {processed_count} documents processed with {error_count} errors.")
                
                # After processing documents, create document nodes
                if processed_count > 0:
                    with st.spinner("Creating document nodes..."):
                        st.session_state.graph = create_document_nodes(
                            st.session_state.documents,
                            st.session_state.graph_client
                        )
                    
                    st.success(f"Processed {processed_count} documents and created document nodes")

    # Tab 2: Document Store Visualization
    with tab2:        
        # Create tabs within Tab 2
        stats_tab, graph_tab = st.tabs(['Document Stats', 'Knowledge Graph'])
        
        with stats_tab:
            if st.button('Show Document Stats'):
                st.subheader('Document Statistics')
                if st.session_state.graph_client is None:
                    st.error("Please connect to Neo4j first!")
                else:
                    try:
                        # Execute Cypher to get document stats
                        docs_query = """
                        MATCH (d:Document) 
                        RETURN count(d) as doc_count
                        """
                        docs_result = st.session_state.graph_client.query(docs_query)
                        
                        # Get document sources
                        sources_query = """
                        MATCH (d:Document)
                        WITH d.source as source, count(*) as count
                        ORDER BY count DESC
                        RETURN source, count
                        """
                        sources_result = st.session_state.graph_client.query(sources_query)
                        
                        if not docs_result or docs_result[0]["doc_count"] == 0:
                            st.warning("No documents found in the database. Please process some documents first.")
                            return
                        
                        # Display document stats
                        doc_count = docs_result[0]["doc_count"]
                        st.write(f"Total documents in database: {doc_count}")
                        
                        # Display source distribution if available
                        if sources_result:
                            st.subheader("Document Sources")
                            source_df = pd.DataFrame(sources_result)
                            st.dataframe(source_df)
                            
                            # Create a simple bar chart of document sources
                            plt.figure(figsize=(10, 6))
                            plt.bar(source_df["source"], source_df["count"])
                            plt.xticks(rotation=45, ha="right")
                            plt.xlabel("Source")
                            plt.ylabel("Count")
                            plt.title("Documents by Source")
                            plt.tight_layout()
                            st.pyplot(plt)
                        
                        # Display sample documents
                        # st.subheader("Sample Documents")
                        # sample_query = """
                        # MATCH (d:Document)
                        # RETURN d.title as title, d.text as text
                        # LIMIT 5
                        # """
                        # samples = st.session_state.graph_client.query(sample_query)
                        
                        # if samples:
                        #     for i, sample in enumerate(samples):
                        #         with st.expander(f"Document: {sample['title']}"):
                        #             st.write(sample["text"])
                        
                    except Exception as e:
                        st.error(f"Error retrieving document stats: {str(e)}")
        
        # Knowledge Graph tab for visualization and similar document connections
        with graph_tab:
            if st.button('Show Knowledge Graph'):
                st.subheader('Knowledge Graph')
                if st.session_state.graph_client is None:
                    st.error("Please connect to Neo4j first!")
                else:
                    # Get visualization
                    plt_fig = visualize_knowledge_graph(st.session_state.graph_client, max_nodes)
                    
                    if plt_fig:
                        st.pyplot(plt_fig)
                        
                        # Get document relationship stats
                        try:
                            rel_query = """
                            MATCH ()-[r:SIMILAR_TO]->()
                            RETURN count(r) as rel_count,
                                avg(r.score) as avg_similarity,
                                min(r.score) as min_similarity,
                                max(r.score) as max_similarity
                            """
                            rel_stats = st.session_state.graph_client.query(rel_query)
                            
                            if rel_stats and rel_stats[0]["rel_count"] > 0:
                                st.write("### Relationship Statistics")
                                st.write(f"Total Relationships: {rel_stats[0]['rel_count']}")
                                st.write(f"Average Similarity: {rel_stats[0]['avg_similarity']:.4f}")
                                st.write(f"Similarity Range: {rel_stats[0]['min_similarity']:.4f} - {rel_stats[0]['max_similarity']:.4f}")
                                
                                # Display the most similar documents
                                st.write("### Most Similar Document Pairs")
                                top_similar_query = """
                                MATCH (d1:Document)-[r:SIMILAR_TO]->(d2:Document)
                                RETURN d1.title as doc1, d2.title as doc2, r.score as similarity
                                ORDER BY r.score DESC
                                LIMIT 10
                                """
                                top_similar = st.session_state.graph_client.query(top_similar_query)
                                
                                if top_similar:
                                    top_similar_df = pd.DataFrame(top_similar)
                                    st.dataframe(top_similar_df)
                        except Exception as e:
                            st.warning(f"Error retrieving relationship statistics: {str(e)}")
                    else:
                        st.warning("No document relationships found or error visualizing graph.")

    # Tab 3: Query the RAG system
    with tab3:
        query = st.text_area(
            "What would you like to know about your documents?",
            height=150,
            placeholder="E.g., 'What are the key features of the recommendation system?'"
        )
        
        # Execute button with conditional search execution
        search_col1, search_col2 = st.columns([1, 4])
        with search_col1:
            search_button = st.button("Search", type="primary", use_container_width=True)
        with search_col2:
            st.write("")  # Spacing
        
        # Execute search when button is clicked
        if query and search_button:
            if st.session_state.vector_store is None:
                st.error("Please configure and connect to Neo4j first!")
            else:
                try:
                    with st.spinner(f"Searching for relevant info from documents..."):
                        # Configure search parameters based on search type
                        search_results = hybrid_search(
                                st.session_state.vector_store,
                                query, 
                                k=top_k,
                                alpha=alpha  # Use the slider value
                            )
                        
                        # Display results
                        if not search_results:
                            st.info("📚 No matching documents found. Try modifying your query or search parameters.")
                        else:
                            st.success(f"Found {len(search_results)} relevant document chunks")
                            
                            # Display results with highlighting and better organization
                            for i, doc in enumerate(search_results):
                                source = doc.metadata.get('source', 'Unknown')
                                # Create a shorter display name for the source
                                display_source = source
                                if len(display_source) > 30:
                                    display_source = display_source[:27] + "..."
                                    
                                # Try to extract page number if available
                                page_info = ""
                                if 'page' in doc.metadata:
                                    page_info = f" (Page {doc.metadata['page']})"
                                    
                                # Format the document content for display
                                content = doc.page_content
                                
                                # If very long, truncate around matched terms
                                if len(content) > context_size * 2:
                                    # Try to find matches of query terms in the content
                                    query_terms = query.lower().split()
                                    found_positions = []
                                    
                                    for term in query_terms:
                                        if len(term) > 3:  # Only match meaningful terms
                                            pos = content.lower().find(term)
                                            if pos >= 0:
                                                found_positions.append(pos)
                                    
                                    # If found matches, center the excerpt around them
                                    if found_positions:
                                        center_pos = sum(found_positions) // len(found_positions)
                                        start = max(0, center_pos - context_size)
                                        end = min(len(content), center_pos + context_size)
                                        
                                        # Add ellipses to show truncation
                                        prefix = "..." if start > 0 else ""
                                        suffix = "..." if end < len(content) else ""
                                        
                                        content = prefix + content[start:end] + suffix
                                
                                # Create expander with clear labeling
                                with st.expander(f"📄 Result {i+1}: {display_source}{page_info}"):
                                    # Document metadata
                                    st.markdown(f"**Source:** {source}")
                                    
                                    # If similarity scores are available
                                    if "score" in doc.metadata:
                                        score = doc.metadata['score']
                                        # Format score as percentage for readability
                                        score_percent = f"{score:.1%}" if score <= 1.0 else f"{score:.2f}"
                                        st.markdown(f"**Relevance Score:** {score_percent}")
                                    
                                    # Document content
                                    st.markdown("**Content:**")
                                    st.markdown(content)
                                    
                                    # Add button to use this specific document for question answering
                                    if st.button(f"Use only this document for answer", key=f"doc_{i}"):
                                        st.session_state.selected_doc = doc.page_content
                            
                            # LLM response based on retrieved documents
                            if generate_answer:
                                groq_api_key = os.getenv("GROQ_API_KEY", "")
                                if not groq_api_key:
                                    st.error("Please set Groq API Key to use the LLM")
                                else:
                                    try:
                                        with st.spinner("Generating AI answer based on retrieved documents..."):
                                            # Handle case where user selected a specific document
                                            if 'selected_doc' in st.session_state:
                                                context = st.session_state.selected_doc
                                                st.session_state.pop('selected_doc', None)  # Clear selection
                                            else:
                                                # Construct context from all search results
                                                context = "\n\n".join([doc.page_content for doc in search_results])
                                            
                                            # Initialize ChatGroq with temperature setting
                                            llm = ChatGroq(
                                                model_name=model_name,
                                                temperature=temperature,
                                                api_key=groq_api_key,
                                            )
                                            
                                            # Construct prompt with improved instructions
                                            prompt = f"""
                                            You are a helpful research assistant that provides detailed answers based on document evidence.
                                            
                                            Answer the following question using ONLY the information in the provided context.
                                            If the context doesn't contain enough information, clearly state what you don't know.
                                            Include specific details and examples from the context when relevant.
                                            Do not make up information that isn't supported by the context.
                                            Always maintain a helpful, informative tone.
                                            
                                            Context:
                                            {context}
                                            
                                            Question: {query}
                                            """
                                            
                                            # Get response
                                            response = llm.invoke([HumanMessage(content=prompt)])
                                            
                                            # Display response in a highlighted box
                                            st.write("### AI Answer")
                                            st.info(response.content)
                                            
                                            # Add to chat history
                                            st.session_state.messages.append({"role": "user", "content": query})
                                            st.session_state.messages.append({"role": "assistant", "content": response.content})
                                            
                                    except Exception as e:
                                        st.error(f"Error generating LLM response: {str(e)}")
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        
        # Display chat history in a collapsible area
        if st.session_state.messages:
            with st.expander("View Chat History", expanded=False):
                st.write("### Previous Queries and Answers")
                
                for i in range(0, len(st.session_state.messages), 2):
                    if i + 1 < len(st.session_state.messages):
                        st.markdown(f"**Query:** {st.session_state.messages[i]['content']}")
                        st.markdown(f"**Answer:** {st.session_state.messages[i+1]['content']}")
                        st.markdown("---")

if __name__ == "__main__":
    main_runner()