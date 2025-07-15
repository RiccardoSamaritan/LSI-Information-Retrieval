import pandas as pd
import spacy
import re

class DataHandler:
    """
    This class provides methods to parse dot-tagged documents into a DataFrame 
    and preprocess text data for LSI (Latent Semantic Indexing).
    """
    @staticmethod
    def parse_to_dataframe(file_path: str) -> pd.DataFrame:
        """
        Parse dot-tagged document file into DataFrame
        Args:
            file_path (str): Path to the document file
        Returns:
            pd.DataFrame: DataFrame containing parsed documents
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        documents = []
        current_doc = {}
        lines = content.split('\n')
        current_field = None
        
        for line in lines:
            if line.startswith('.I'):
                if current_doc:
                    documents.append(current_doc)
                current_doc = {'I': line[3:].strip()}
                current_field = None
            elif line.startswith('.T'):
                current_field = 'T'
                current_doc[current_field] = ''
            elif line.startswith('.A'):
                current_field = 'A'
                current_doc[current_field] = ''
            elif line.startswith('.B'):
                current_field = 'B'
                current_doc[current_field] = ''
            elif line.startswith('.W'):
                current_field = 'W'
                current_doc[current_field] = ''
            elif current_field:
                current_doc[current_field] += line + ' '
        
        if current_doc:
            documents.append(current_doc)

        for doc in documents:
            for key in doc:
                doc[key] = doc[key].strip()
        
        return pd.DataFrame(documents)
    
    @staticmethod
    def preprocess_for_lsi(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess text data for LSI
        Args:
            df (pd.DataFrame): DataFrame containing documents
            **kwargs: Additional parameters for preprocessing
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        text_columns = kwargs.get('text_columns', ['W'])
        lowercase = kwargs.get('lowercase', True)
        remove_punct = kwargs.get('remove_punct', True)
        remove_stop = kwargs.get('remove_stop', True)
        lemmatize = kwargs.get('lemmatize', True)
        remove_num = kwargs.get('remove_num', True)
        allowed_pos = kwargs.get('allowed_pos', ['NOUN', 'PROPN', 'ADJ', 'VERB', 'ADV'])

        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        
        processed_df = df.copy()

        combined_text = []
        for _, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    text_parts.append(str(row[col]))
            combined_text.append(' '.join(text_parts))

        clean_texts = []
        for text in combined_text:
            if remove_punct and remove_num:
                text = re.sub(r'[^a-zA-Z ]+', '', text)
            elif remove_punct:
                text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)

            doc = nlp(text)
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
                
                if word.strip() and len(word) > 2:  # Remove very short words
                    tokens.append(word)
            
            clean_texts.append(' '.join(tokens))
        
        processed_df['clean_text'] = clean_texts
        return processed_df