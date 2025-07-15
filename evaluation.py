import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from lsi import LSI

@dataclass
class EvaluationResult:
    """
    Store evaluation metrics for a query
    """
    query_id: int
    query_text: str
    retrieved_docs: List[int]
    relevant_docs: List[int]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    average_precision: float
    reciprocal_rank: float

class EvaluationFramework:
    """
    Comprehensive evaluation of the LSI system
    """
    
    def __init__(self, system: LSI):
        self.system = system
        self.queries = {}
        self.relevance_judgments = {}
        
    def load_time_queries(self, query_file: str, rel_file: str):
        """
        Load TIME dataset queries and relevance judgments
        Args:
            query_file (str): Path to the query file
            rel_file (str): Path to the relevance judgments file
        """
        with open(rel_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    query_id = int(parts[0])
                    relevant_docs = [int(x) for x in parts[1:]]
                    self.relevance_judgments[query_id] = relevant_docs
        
        sample_queries = {
            1: "economic policy government spending",
            2: "space exploration moon landing",
            3: "environmental pollution climate change",
            4: "technology computer artificial intelligence",
            5: "medical research cancer treatment",
            6: "education system school reform",
            7: "international relations foreign policy",
            8: "business economics market trends",
            9: "social issues civil rights",
            10: "science research development"
        }
        
        self.queries = sample_queries
        
    def compute_precision_recall(self, retrieved: List[int], relevant: List[int], k_values: List[int]) -> Tuple[Dict, Dict]:
        """
        Compute precision and recall at different k values
        Args:
            retrieved (List[int]): List of retrieved document IDs
            relevant (List[int]): List of relevant document IDs
            k_values (List[int]): List of k values to compute precision/recall for
        Returns:
            Tuple[Dict, Dict]: Precision and recall at k values
        """
        precision_at_k = {}
        recall_at_k = {}
        
        for k in k_values:
            retrieved_k = retrieved[:k]
            relevant_retrieved = len(set(retrieved_k) & set(relevant))
            
            precision_at_k[k] = relevant_retrieved / k if k > 0 else 0
            recall_at_k[k] = relevant_retrieved / len(relevant) if len(relevant) > 0 else 0
        
        return precision_at_k, recall_at_k
    
    def compute_average_precision(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        Compute Average Precision (AP)
        Args:
            retrieved (List[int]): List of retrieved document IDs
            relevant (List[int]): List of relevant document IDs
        Returns:
            float: Average Precision score
        """
        if not relevant:
            return 0.0
        
        ap = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap += precision_at_i
        
        return ap / len(relevant)
    
    def compute_reciprocal_rank(self, retrieved: List[int], relevant: List[int]) -> float:
        """
        Compute Reciprocal Rank (RR)
        Args:
            retrieved (List[int]): List of retrieved document IDs
            relevant (List[int]): List of relevant document IDs
        Returns:
            float: Reciprocal Rank score
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def evaluate_single_query(self, query_id: int, k_values: List[int] = [5, 10, 20]) -> EvaluationResult:
        """
        Evaluate system performance for a single query
        Args:
            query_id (int): The ID of the query to evaluate
            k_values (List[int]): List of k values for precision/recall computation
        Returns:
            EvaluationResult: Evaluation metrics for the query
        """
        if query_id not in self.queries:
            raise ValueError(f"Query {query_id} not found")
        
        query_text = self.queries[query_id]
        relevant_docs = self.relevance_judgments.get(query_id, [])

        retrieved_indices, _ = self.system.retrieve(query_text, max(k_values))
        
        retrieved_docs = [int(self.system.parsed_df.iloc[idx]['I']) for idx in retrieved_indices]
     
        precision_at_k, recall_at_k = self.compute_precision_recall(retrieved_docs, relevant_docs, k_values)
        avg_precision = self.compute_average_precision(retrieved_docs, relevant_docs)
        reciprocal_rank = self.compute_reciprocal_rank(retrieved_docs, relevant_docs)
        
        return EvaluationResult(
            query_id=query_id,
            query_text=query_text,
            retrieved_docs=retrieved_docs,
            relevant_docs=relevant_docs,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            average_precision=avg_precision,
            reciprocal_rank=reciprocal_rank
        )
    
    def evaluate_all_queries(self, k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate system performance across all queries
        Args:
            k_values (List[int]): List of k values for precision/recall computation
        Returns:
            Dict[str, float]: Aggregate evaluation metrics
        """
        results = []
        
        print("Evaluating queries...")
        for query_id in self.queries.keys():
            if query_id in self.relevance_judgments:
                result = self.evaluate_single_query(query_id, k_values)
                results.append(result)
                print(f"Query {query_id}: MAP={result.average_precision:.3f}, MRR={result.reciprocal_rank:.3f}")
        
        metrics = {}
        
        metrics['MAP'] = np.mean([r.average_precision for r in results])
        
        metrics['MRR'] = np.mean([r.reciprocal_rank for r in results])
        
        for k in k_values:
            metrics[f'P@{k}'] = np.mean([r.precision_at_k[k] for r in results])
            metrics[f'R@{k}'] = np.mean([r.recall_at_k[k] for r in results])
        
        return metrics, results