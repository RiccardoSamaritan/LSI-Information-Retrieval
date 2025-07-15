import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from dataprocessor import DataHandler
from sys_config import SystemConfiguration
from term_document_matrix import TermDocumentMatrix
from queryprocessor import QueryProcessor
from lsi_core import LSICore

class LSI:
    """
    Latent Semantic Indexing (LSI) System
    """
    def __init__(self, data_path: str, config: SystemConfiguration):
        self.data_path = data_path
        self.config = config
        self.setup_system()
    
    def setup_system(self):
        """
        Set up the LSI system by parsing documents, preprocessing text,
        building the term-document matrix, and computing LSI decomposition.
        This includes initializing the LSI core and preparing the IDF weights for query processing.
        """
        print("Setting up LSI-IR System...")
        print(f"Configuration: {self.config.n_components} components, {self.config.metric} metric")

        print("1. Parsing documents...")
        self.parsed_df = DataHandler.parse_to_dataframe(self.data_path)
        print(f"   Loaded {len(self.parsed_df)} documents")
        
        print("2. Preprocessing documents...")
        self.preprocessed_df = DataHandler.preprocess_for_lsi(
            self.parsed_df, **self.config.preprocessing_config
        )

        print("3. Building term-document matrix...")
        self.term_document_matrix, self.term_indexes = TermDocumentMatrix.build_matrix(
            self.preprocessed_df, self.config.metric
        )
        print(f"   Matrix shape: {self.term_document_matrix.shape}")

        print("4. Computing LSI decomposition...")
        self.document_indexes = np.array(self.parsed_df["T"])
        self.lsi = LSICore(
            self.term_document_matrix, 
            self.config.n_components,
            self.document_indexes,
            self.term_indexes
        )

        doc_freq = np.count_nonzero(self.term_document_matrix, axis=1)
        n_docs = self.term_document_matrix.shape[1]
        self.idf_weights = np.log10(n_docs / (doc_freq + 1))
        
        print("5. System ready!")
        print(f"   Vocabulary size: {len(self.term_indexes)}")
        print(f"   LSI dimensions: {self.config.n_components}")
    
    def retrieve(self, query: str, n_docs: int = 10) -> List[int]:
        """
        Retrieve documents based on a query using LSI.
        Args:
            query (str): The query string to process
            n_docs (int): Number of documents to retrieve
        Returns:
            List[int]: Indices of the top retrieved documents
        """
        preprocessed_query = QueryProcessor.preprocess_query(
            query, **self.config.preprocessing_config
        )
        
        query_vector = QueryProcessor.create_query_vector(
            preprocessed_query, self.term_indexes, "freq"
        )

        if self.config.metric == "tf-idf":
            query_vector = query_vector * self.idf_weights

        query_lsi = np.linalg.inv(np.diag(self.lsi.concept_strength)) @ \
                   self.lsi.term_concept_similarity.T @ query_vector.reshape(-1, 1)

        similarities = cosine_similarity(
            query_lsi.T, 
            self.lsi.document_concept_similarity
        )[0]

        top_indices = np.argsort(similarities)[::-1][:n_docs]
        
        return top_indices, similarities[top_indices]
    
    def display_results(self, query: str, doc_indices: List[int], similarities: List[float], n_display: int = 5):
        """
        Display the top retrieved documents for a given query.
        Args:
            query (str): The original query string
            doc_indices (List[int]): Indices of the retrieved documents
            similarities (List[float]): Similarity scores for the retrieved documents
            n_display (int): Number of documents to display
        """
        print(f"\n=== Query: {query} ===")
        print(f"Top {min(n_display, len(doc_indices))} results:")
        
        for i, (doc_idx, sim) in enumerate(zip(doc_indices[:n_display], similarities[:n_display])):
            title = self.document_indexes[doc_idx]
            print(f"{i+1}. Doc {doc_idx}: {title}")
            print(f"   Similarity: {sim:.4f}")
            if 'W' in self.parsed_df.columns:
                content = self.parsed_df.iloc[doc_idx]['W'][:200] + "..."
                print(f"   Content: {content}")
            print()