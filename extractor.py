import re
import pandas as pd
from typing import List, Dict, Any, Union, Optional
import logging
from preprocess import DataPreprocessor

logger = logging.getLogger(__name__)

class TransactionExtractor:
    """Extracts transaction records from preprocessed data."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.preprocessor = DataPreprocessor()
        
        # Common column mappings for structured data
        self.column_mappings = {
            'date': ['date', 'transaction_date', 'trans_date', 'posting_date', 'effective_date'],
            'description': ['description', 'desc', 'memo', 'details', 'transaction_details', 'payee'],
            'amount': ['amount', 'transaction_amount', 'trans_amount'],
            'debit': ['debit', 'debit_amount', 'withdrawal', 'outgoing', 'payment'],
            'credit': ['credit', 'credit_amount', 'deposit', 'incoming', 'receipt'],
            'balance': ['balance', 'running_balance', 'account_balance'],
            'reference': ['reference', 'ref', 'transaction_id', 'check_number'],
        }
    
    def extract_from_structured_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract transactions from structured data (CSV/Excel).
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            List of transaction dictionaries
        """
        self.logger.info(f"Extracting transactions from structured data ({len(df)} rows)")
        
        # Map columns to standard names
        column_map = self._map_columns(df.columns.tolist())
        
        transactions = []
        for idx, row in df.iterrows():
            try:
                transaction = self._extract_transaction_from_row(row, column_map)
                if transaction:
                    transactions.append(transaction)
            except Exception as e:
                self.logger.warning(f"Error processing row {idx}: {str(e)}")
                continue
        
        self.logger.info(f"Extracted {len(transactions)} transactions from structured data")
        return transactions
    
    def extract_from_text_data(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Extract transactions from text data (PDF/DOCX).
        
        Args:
            lines: List of preprocessed text lines
            
        Returns:
            List of transaction dictionaries
        """
        self.logger.info(f"Extracting transactions from text data ({len(lines)} lines)")
        
        transactions = []
        for line_num, line in enumerate(lines):
            try:
                transaction = self._extract_transaction_from_line(line)
                if transaction:
                    transactions.append(transaction)
            except Exception as e:
                self.logger.warning(f"Error processing line {line_num}: {str(e)}")
                continue
        
        self.logger.info(f"Extracted {len(transactions)} transactions from text data")
        return transactions
    
    def _map_columns(self, columns: List[str]) -> Dict[str, str]:
        """Map DataFrame columns to standard field names."""
        column_map = {}
        columns_lower = [col.lower().strip() for col in columns]
        
        for standard_field, possible_names in self.column_mappings.items():
            for col_idx, col_name in enumerate(columns_lower):
                if any(possible_name in col_name for possible_name in possible_names):
                    column_map[standard_field] = columns[col_idx]  # Use original case
                    break
        
        self.logger.info(f"Column mapping: {column_map}")
        return column_map
    
    def _extract_transaction_from_row(self, row: pd.Series, column_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Extract transaction data from a DataFrame row."""
        # Get mapped values
        date_val = row.get(column_map.get('date', ''), '')
        description_val = row.get(column_map.get('description', ''), '')
        amount_val = row.get(column_map.get('amount', ''), 0)
        debit_val = row.get(column_map.get('debit', ''), 0)
        credit_val = row.get(column_map.get('credit', ''), 0)
        reference_val = row.get(column_map.get('reference', ''), '')
        
        # Clean and normalize values
        transaction_date = self.preprocessor.normalize_date(str(date_val))
        if not transaction_date:
            return None
        
        # Determine amounts
        if amount_val and not debit_val and not credit_val:
            # Single amount column - determine if debit or credit based on sign
            clean_amount = self.preprocessor.clean_amount(amount_val)
            if clean_amount < 0:
                debit_amount = abs(clean_amount)
                credit_amount = 0.0
            else:
                debit_amount = 0.0
                credit_amount = clean_amount
        else:
            debit_amount = self.preprocessor.clean_amount(debit_val)
            credit_amount = self.preprocessor.clean_amount(credit_val)
        
        # Skip if no meaningful amount
        if debit_amount == 0 and credit_amount == 0:
            return None
        
        # Build transaction record
        description_clean = str(description_val).strip() if description_val else "Unknown Transaction"
        
        transaction = {
            'raw_description': description_clean,
            'reference': str(reference_val).strip() if reference_val else '',
            'debit_amount': debit_amount,
            'credit_amount': credit_amount,
            'transaction_date': transaction_date,
            'source_data': 'structured'
        }
        
        return transaction
    
    def _extract_transaction_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract transaction data from a text line."""
        # Pattern for common bank statement line formats
        # Typical format: DATE DESCRIPTION AMOUNT
        patterns = [
            # Pattern 1: Date at start, amount at end
            r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\s+(.+?)\s+([-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*$',
            # Pattern 2: Date, description, debit, credit
            r'(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})\s+(.+?)\s+([-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s+([-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            # Pattern 3: More flexible pattern
            r'(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})\s+(.+?)\s+([-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                groups = match.groups()
                
                if len(groups) >= 3:
                    date_str = groups[0]
                    description = groups[1].strip()
                    amount_str = groups[2]
                    
                    # Normalize date
                    transaction_date = self.preprocessor.normalize_date(date_str)
                    if not transaction_date:
                        continue
                    
                    # Clean amount
                    amount = self.preprocessor.clean_amount(amount_str)
                    if amount == 0:
                        continue
                    
                    # Determine if debit or credit
                    if amount < 0 or '-' in amount_str:
                        debit_amount = abs(amount)
                        credit_amount = 0.0
                    else:
                        debit_amount = 0.0
                        credit_amount = amount
                    
                    # Handle case with separate debit/credit columns
                    if len(groups) >= 4:
                        credit_str = groups[3]
                        credit_amount = self.preprocessor.clean_amount(credit_str)
                        debit_amount = amount  # First amount is debit
                    
                    return {
                        'raw_description': description,
                        'reference': '',
                        'debit_amount': debit_amount,
                        'credit_amount': credit_amount,
                        'transaction_date': transaction_date,
                        'source_data': 'text'
                    }
        
        return None