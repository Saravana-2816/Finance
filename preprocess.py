import re
import pandas as pd
from typing import List, Dict, Any, Union
from datetime import datetime
import logging
from dateutil import parser

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles preprocessing of extracted data from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',           # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',           # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',           # MM-DD-YYYY
            r'\d{2}\.\d{2}\.\d{4}',         # MM.DD.YYYY
            r'\d{1,2}/\d{1,2}/\d{4}',       # M/D/YYYY
            r'\d{1,2}-\d{1,2}-\d{4}',       # M-D-YYYY
        ]
        
        self.amount_pattern = r'[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        
    def preprocess_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess structured data (CSV/Excel).
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Preprocessing structured data with {len(df)} rows")
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Convert column names to lowercase and strip whitespace
        df.columns = df.columns.astype(str).str.lower().str.strip()
        
        # Remove rows that appear to be headers (repeated column names)
        header_mask = df.apply(
            lambda row: any(str(val).lower().strip() in df.columns for val in row if pd.notna(val)), 
            axis=1
        )
        df = df[~header_mask]
        
        # Identify potential transaction rows
        df = self._identify_transaction_rows(df)
        
        # Clean and standardize data types
        df = self._clean_structured_data(df)
        
        self.logger.info(f"After preprocessing: {len(df)} rows remain")
        return df
    
    def preprocess_text_data(self, text_data: Union[str, List[str]]) -> List[str]:
        """
        Preprocess unstructured text data (PDF/DOCX).
        
        Args:
            text_data: Raw text or list of text blocks
            
        Returns:
            List of cleaned text lines
        """
        if isinstance(text_data, str):
            text_data = [text_data]
        
        all_lines = []
        for text_block in text_data:
            lines = text_block.split('\n')
            all_lines.extend(lines)
        
        # Clean and filter lines
        cleaned_lines = []
        for line in all_lines:
            line = line.strip()
            if self._is_potential_transaction_line(line):
                cleaned_lines.append(line)
        
        self.logger.info(f"Preprocessed text data: {len(cleaned_lines)} potential transaction lines")
        return cleaned_lines
    
    def _identify_transaction_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify rows that likely contain transaction data."""
        # Look for rows with date and amount patterns
        transaction_mask = pd.Series([False] * len(df))
        
        for col in df.columns:
            # Check for date patterns
            date_mask = df[col].astype(str).str.contains('|'.join(self.date_patterns), na=False)
            
            # Check for amount patterns
            amount_mask = df[col].astype(str).str.contains(self.amount_pattern, na=False)
            
            # Row is likely a transaction if it has both date and amount
            transaction_mask |= (date_mask & df.apply(
                lambda row: any(re.search(self.amount_pattern, str(val)) for val in row), axis=1
            ))
        
        return df[transaction_mask]
    
    def _clean_structured_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean structured data types and formats."""
        # Remove currency symbols and commas from potential amount columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains amounts
                sample_values = df[col].dropna().astype(str).head(10)
                if any(re.search(self.amount_pattern, val) for val in sample_values):
                    df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    
    def _is_potential_transaction_line(self, line: str) -> bool:
        """Determine if a text line potentially contains transaction data."""
        if not line or len(line.strip()) < 5:  # Reduced minimum length
            return False
        
        line_lower = line.lower().strip()
        
        # Skip obvious header/footer patterns
        skip_patterns = [
            r'^page \d+',
            r'^statement period',
            r'^account summary',
            r'^balance forward',
            r'^opening balance',
            r'^closing balance',
            r'^beginning balance',
            r'^ending balance',
            r'^total',
            r'^\*+',
            r'^-+']
    
    def normalize_date(self, date_str: str) -> str:
        """
        Normalize date string to YYYY-MM-DD format.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Normalized date string in YYYY-MM-DD format
        """
        if not date_str or pd.isna(date_str):
            return ""
        
        try:
            # Try parsing with dateutil parser first (most flexible)
            parsed_date = parser.parse(str(date_str), fuzzy=True)
            return parsed_date.strftime('%Y-%m-%d')
        except:
            # If that fails, try manual pattern matching
            for pattern in self.date_patterns:
                match = re.search(pattern, str(date_str))
                if match:
                    date_part = match.group()
                    try:
                        if '-' in date_part:
                            parts = date_part.split('-')
                        elif '/' in date_part:
                            parts = date_part.split('/')
                        elif '.' in date_part:
                            parts = date_part.split('.')
                        else:
                            continue
                        
                        if len(parts) == 3:
                            # Determine format based on first part length
                            if len(parts[0]) == 4:  # YYYY-MM-DD
                                return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                            else:  # MM-DD-YYYY or DD-MM-YYYY
                                year = parts[2] if len(parts[2]) == 4 else f"20{parts[2]}"
                                return f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                    except:
                        continue
        
        self.logger.warning(f"Could not normalize date: {date_str}")
        return ""
    
    def clean_amount(self, amount_str: Union[str, float, int]) -> float:
        """
        Clean and normalize monetary amounts.
        
        Args:
            amount_str: Amount in various formats
            
        Returns:
            Cleaned float amount
        """
        if pd.isna(amount_str) or amount_str == "":
            return 0.0
        
        if isinstance(amount_str, (int, float)):
            return float(amount_str)
        
        # Remove currency symbols, commas, and whitespace
        cleaned = re.sub(r'[^\d.-]', '', str(amount_str).strip())
        
        if not cleaned or cleaned == '-':
            return 0.0
        
        try:
            return float(cleaned)
        except ValueError:
            self.logger.warning(f"Could not parse amount: {amount_str}")
            return 0.0