import numpy as np
import spacy
import re

class QueryProcessor:
    """
    This class handles query preprocessing and vector creation
    """
    @staticmethod
    def preprocess_query(query: str, **kwargs) -> str:
        """
        Preprocess query using same pipeline as documents
        Args:
            query (str): The query string to preprocess
            **kwargs: Additional parameters for preprocessing
        Returns:
            str: Preprocessed query string
        """
        lowercase = kwargs.get('lowercase', True)
        remove_punct = kwargs.get('remove_punct', True)
        remove_stop = kwargs.get('remove_stop', True)
        lemmatize = kwargs.get('lemmatize', True)
        remove_num = kwargs.get('remove_num', True)
        allowed_pos = kwargs.get('allowed_pos', ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'])

        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

        if remove_punct and remove_num:
            query = re.sub(r'[^a-zA-Z ]+', '', query)
        elif remove_punct:
            query = re.sub(r'[^a-zA-Z0-9 ]+', '', query)
        
        doc = nlp(query)
        tokens = []
        
        for token in doc:
            if remove_punct and token.is_punct:
                continue
            if remove_stop and token.is_stop:
                continue
            if remove_num and token.like_num:
                continue
            if allowed_pos and token.pos_ not in allowed_pos:
                continue
            
            word = token.lemma_ if lemmatize else token.text
            word = word.lower() if lowercase else word
            
            if remove_punct:
                word = re.sub(r'[^a-zA-Z]+', '', word)
            if remove_num:
                word = re.sub(r'[0-9]+', '', word)
            
            if word.strip() and len(word) > 2:
                tokens.append(word)
        
        return ' '.join(tokens)
    
    @staticmethod
    def create_query_vector(preprocessed_query: str, term_indexes: np.ndarray, metric: str = "freq") -> np.ndarray:
        """
        Create term vector for query
        Args:
            preprocessed_query (str): Preprocessed query string
            term_indexes (np.ndarray): Vocabulary indexes
            metric (str): Metric for the vector ('bool', 'freq')
        Returns:
            np.ndarray: Query vector
        """
        query_vector = np.zeros(len(term_indexes))
        
        for term in preprocessed_query.split():
            idx = np.where(term_indexes == term)[0]
            if len(idx) > 0:
                query_vector[idx[0]] += 1
                if metric == "bool":
                    query_vector[idx[0]] = 1
        
        return query_vector