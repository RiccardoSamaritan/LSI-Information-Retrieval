import pandas as pd
from typing import List
from matplotlib import pyplot as plt

from sys_config import SystemConfiguration
from lsi import LSI
from evaluation import EvaluationFramework

class DimensionEvaluator:
    """
    Evaluate system performance across different LSI dimensions
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def evaluate_dimensions(self, dimension_range: List[int], 
                          query_file: str = None, rel_file: str = None) -> pd.DataFrame:
        """
        Evaluate performance across different LSI dimensions
        Args:
            dimension_range (List[int]): List of dimensions to evaluate
            query_file (str): Path to query file (optional)
            rel_file (str): Path to relevance file (optional)
        Returns:
            pd.DataFrame: DataFrame containing evaluation results for each dimension
        """
        results = []
        
        for n_components in dimension_range:
            print(f"\n=== Evaluating {n_components} dimensions ===")

            config = SystemConfiguration(n_components=n_components)
            system = LSI(self.data_path, config)

            evaluator = EvaluationFramework(system)
            if query_file and rel_file:
                evaluator.load_time_queries(query_file, rel_file)
            else:
                evaluator.load_time_queries("", "")
            
            metrics, _ = evaluator.evaluate_all_queries()

            result_row = {'n_components': n_components}
            result_row.update(metrics)
            results.append(result_row)
            
            print(f"Results: MAP={metrics['MAP']:.3f}, MRR={metrics['MRR']:.3f}")
        
        return pd.DataFrame(results)
    
    def plot_dimension_analysis(self, results_df: pd.DataFrame):
        """
        Plot evaluation results across different dimensions
        Args:
            results_df (pd.DataFrame): DataFrame containing evaluation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(results_df['n_components'], results_df['MAP'], 'o-', color='blue')
        axes[0, 0].set_title('Mean Average Precision vs Dimensions')
        axes[0, 0].set_xlabel('Number of LSI Dimensions')
        axes[0, 0].set_ylabel('MAP')
        axes[0, 0].grid(True)

        axes[0, 1].plot(results_df['n_components'], results_df['MRR'], 'o-', color='red')
        axes[0, 1].set_title('Mean Reciprocal Rank vs Dimensions')
        axes[0, 1].set_xlabel('Number of LSI Dimensions')
        axes[0, 1].set_ylabel('MRR')
        axes[0, 1].grid(True)

        k_values = [5, 10, 20]
        for k in k_values:
            if f'P@{k}' in results_df.columns:
                axes[1, 0].plot(results_df['n_components'], results_df[f'P@{k}'], 
                               'o-', label=f'P@{k}')
        axes[1, 0].set_title('Precision@K vs Dimensions')
        axes[1, 0].set_xlabel('Number of LSI Dimensions')
        axes[1, 0].set_ylabel('Precision@K')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        for k in k_values:
            if f'R@{k}' in results_df.columns:
                axes[1, 1].plot(results_df['n_components'], results_df[f'R@{k}'], 
                               'o-', label=f'R@{k}')
        axes[1, 1].set_title('Recall@K vs Dimensions')
        axes[1, 1].set_xlabel('Number of LSI Dimensions')
        axes[1, 1].set_ylabel('Recall@K')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()