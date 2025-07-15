import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict

class TermDocumentMatrix:
    """
    Handle term-document matrix creation and operations
    """
    @staticmethod
    def create_vocabulary(docs: pd.Series) -> Tuple[List[str], List[Dict]]:
        """
        Create vocabulary and term frequencies
        Args:
            docs (pd.Series): Series containing document texts
        Returns:
            Tuple[List[str], List[Dict]]: Vocabulary list and term frequencies for each document
        """
        vocab = []
        term_frequencies = []
        
        for doc in docs:
            tokens = doc.split()
            doc_tf = Counter(tokens)
            term_frequencies.append(doc_tf)
            vocab.extend(list(set(tokens)))
        
        return list(set(vocab)), term_frequencies
    
    @staticmethod
    def build_matrix(df: pd.DataFrame, metric: str = "freq") -> Tuple[np.ndarray, np.ndarray]:
        """
        Build term-document matrix from DataFrame
        Args:
            df (pd.DataFrame): DataFrame containing documents
            metric (str): Metric for the matrix ('bool', 'freq', 'tf-idf')
        Returns:
            Tuple[np.ndarray, np.ndarray]: Term-document matrix and vocabulary indexes
        """
        if metric not in ["bool", "freq", "tf-idf"]:
            raise ValueError("Invalid metric. Choose from: bool, freq, tf-idf")
        
        docs = df["clean_text"]
        vocab, term_freq = TermDocumentMatrix.create_vocabulary(docs)
        
        sorted_vocab = sorted(vocab)
        indexed_vocab = {term: idx for idx, term in enumerate(sorted_vocab)}
        
        term_document_matrix = np.zeros((len(vocab), len(docs)))
        
        for doc_idx, tf_doc in enumerate(term_freq):
            for term, freq in tf_doc.items():
                if metric == "bool":
                    freq = 1
                term_document_matrix[indexed_vocab[term], doc_idx] = freq
        
        if metric == "tf-idf":
            term_document_matrix = TermDocumentMatrix.apply_tfidf(term_document_matrix)
        return term_document_matrix, np.array(sorted_vocab)
    
    @staticmethod
    def apply_tfidf(td_matrix: np.ndarray) -> np.ndarray:
        """
        Apply TF-IDF weighting to the term-document matrix
        Args:
            td_matrix (np.ndarray): Term-document matrix
        Returns:
            np.ndarray: TF-IDF weighted term-document matrix
        """
        doc_freq = np.count_nonzero(td_matrix, axis=1)
        n_docs = td_matrix.shape[1]
        idf = np.log10(n_docs / (doc_freq + 1))
        
        tfidf_matrix = td_matrix.copy()
        for i in range(td_matrix.shape[0]):
            tfidf_matrix[i, :] *= idf[i]
        
        return tfidf_matrix