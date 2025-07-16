import pandas as pd
import spacy
import re
import os

class DataHandler:
    """
    This class provides methods to parse dot-tagged documents into a DataFrame 
    and preprocess text data for Latent Semantic Indexing.
    """
    @staticmethod
    def load_stopwords(stopwords_file: str = "TIME.STP") -> set:
        """
        Load stopwords from .STP file
        Args:
            stopwords_file (str): Path to stopwords file
        Returns:
            set: Set of stopwords
        """
        stopwords = set()
        if os.path.exists(stopwords_file):
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            stopwords.add(word)
            except Exception as e:
                print(f"   Warning: Could not load stopwords from {stopwords_file}: {e}")
                print("   Using spaCy default stopwords")
        else:
            print(f"   Warning: Stopwords file {stopwords_file} not found")
            print("   Using spaCy default stopwords")
        
        return stopwords
    
    @staticmethod
    def parse_to_dataframe(file_path: str) -> pd.DataFrame:
        """
        Parse TIME dataset file into DataFrame
        The TIME.ALL file uses a specific format with *TEXT markers
        Args:
            file_path (str): Path to the TIME.ALL file
        Returns:
            pd.DataFrame: DataFrame containing parsed documents
        """
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        documents = []

        text_sections = content.split('*TEXT')
        
        for section in text_sections:
            section = section.strip()
            if not section:
                continue
   
            lines = section.split('\n')
            
            doc_id = None
            title = ""
            content_lines = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
         
                if i < 3 and any(char.isdigit() for char in line):
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        doc_id = int(parts[0])
                        continue
      
                if any(keyword in line.upper() for keyword in ['PAGE', 'TEXT']):
                    continue

                if not title and len(line) > 10:
                    title = line
                    continue

                content_lines.append(line)
            
            if doc_id is not None:
                doc_content = ' '.join(content_lines)
                
                document = {
                    'I': doc_id,
                    'T': title,
                    'A': '', 
                    'B': '',
                    'W': doc_content
                }
                documents.append(document)
        
        df = pd.DataFrame(documents)
        
        for col in ['I', 'T', 'A', 'B', 'W']:
            if col not in df.columns:
                df[col] = ''

        if 'I' in df.columns:
            df['I'] = pd.to_numeric(df['I'], errors='coerce').fillna(0).astype(int)
        
        return df
    
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

        custom_stopwords = set()
        if remove_stop:
            custom_stopwords = DataHandler.load_stopwords()
        
        processed_df = df.copy()

        combined_text = []
        for _, row in df.iterrows():
            text_parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(str(row[col]))
            combined_text.append(' '.join(text_parts))

        clean_texts = []
        for text in combined_text:
            if not text.strip():
                clean_texts.append('')
                continue
      
            if remove_punct and remove_num:
                text = re.sub(r'[^a-zA-Z ]+', ' ', text)
            elif remove_punct:
                text = re.sub(r'[^a-zA-Z0-9 ]+', ' ', text)

            doc = nlp(text)
            tokens = []
            
            for token in doc:
                if remove_punct and token.is_punct:
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
                
                if not word.strip() or len(word) <= 2:
                    continue
            
                if remove_stop:
                    if token.is_stop or word.lower() in custom_stopwords:
                        continue
                
                tokens.append(word)
            
            clean_texts.append(' '.join(tokens))
        
        processed_df['clean_text'] = clean_texts
        return processed_df