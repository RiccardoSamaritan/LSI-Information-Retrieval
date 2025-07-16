import numpy as np
from typing import Optional
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

class LSICore:
    """
    Core LSI operations including SVD decomposition and concept analysis
    """
    
    def __init__(self, term_document_matrix: np.ndarray, n_components: int = 100, 
                 document_indexes: Optional[np.ndarray] = None, 
                 term_indexes: Optional[np.ndarray] = None):
        self.term_document_matrix = term_document_matrix
        self.n_components = min(n_components, min(term_document_matrix.shape) - 1)
        self.document_indexes = document_indexes
        self.term_indexes = term_indexes
        self.compute_lsi()
    
    def compute_lsi(self):
        """
        Perform SVD decomposition on the term-document matrix. Random state is set for reproducibility.
        """
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.term_concept_similarity = svd.fit_transform(self.term_document_matrix)
        self.term_concept_similarity = self.term_concept_similarity / svd.singular_values_
        self.concept_strength = svd.singular_values_
        self.document_concept_similarity = svd.components_.T
        self.svd_model = svd
    
    def analyze_concepts(self, concept_index: int = 0, n_terms: int = 20):
        """
        Analyze term weights for a concept by plotting the top terms.
        Args:
            concept_index (int): Index of the concept to analyze
            n_terms (int): Number of terms to display
        """
        if concept_index >= self.n_components:
            raise ValueError(f"Concept index must be < {self.n_components}")
        
        concept_weights = np.abs(self.term_concept_similarity[:, concept_index])
        top_indices = np.argsort(concept_weights)[-n_terms:][::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(n_terms), 
                self.term_concept_similarity[top_indices, concept_index][::-1],
                color='orange')
        
        if self.term_indexes is not None:
            plt.yticks(range(n_terms), self.term_indexes[top_indices][::-1])
        
        plt.title(f"Term Weights for Concept {concept_index}")
        plt.xlabel("Weight")
        plt.tight_layout()
        plt.show()