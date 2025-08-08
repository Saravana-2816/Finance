import json
import re
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from pydantic import BaseModel

import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import tempfile
from dateutil import parser as date_parser

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class Transaction(BaseModel):
    date: str
    amount: float
    transaction_type: str  # Credit or Debit
    merchant_name: str
    category: str
    subcategory: str
    account_involved: str
    location: Optional[str] = None
    description: str

# Merchant Database
class MerchantDatabase:
    def __init__(self):
        self.merchants = {
            # Food & Dining
            "McDonald's": {"category": "Food", "subcategory": "Fast Food"},
            "KFC": {"category": "Food", "subcategory": "Fast Food"},
            "Starbucks": {"category": "Food", "subcategory": "Coffee"},
            "Pizza Hut": {"category": "Food", "subcategory": "Pizza"},
            "Domino's": {"category": "Food", "subcategory": "Pizza"},
            "Subway": {"category": "Food", "subcategory": "Fast Food"},
            "Burger King": {"category": "Food", "subcategory": "Fast Food"},
            
            # Transportation
            "Uber": {"category": "Transportation", "subcategory": "Ride Sharing"},
            "Lyft": {"category": "Transportation", "subcategory": "Ride Sharing"},
            "Air India": {"category": "Transportation", "subcategory": "Airlines"},
            "Emirates": {"category": "Transportation", "subcategory": "Airlines"},
            "Delta": {"category": "Transportation", "subcategory": "Airlines"},
            "Ola": {"category": "Transportation", "subcategory": "Ride Sharing"},
            
            # Shopping
            "Amazon": {"category": "Shopping", "subcategory": "Online Retail"},
            "Walmart": {"category": "Shopping", "subcategory": "Grocery"},
            "Target": {"category": "Shopping", "subcategory": "Department Store"},
            "Flipkart": {"category": "Shopping", "subcategory": "Online Retail"},
            "BigBasket": {"category": "Shopping", "subcategory": "Grocery"},
            
            # Utilities
            "BSES": {"category": "Utilities", "subcategory": "Electricity"},
            "Airtel": {"category": "Utilities", "subcategory": "Telecom"},
            "Jio": {"category": "Utilities", "subcategory": "Telecom"},
            "TATA Power": {"category": "Utilities", "subcategory": "Electricity"},
            
            # Entertainment
            "Netflix": {"category": "Entertainment", "subcategory": "Streaming"},
            "Spotify": {"category": "Entertainment", "subcategory": "Music"},
            "Disney+": {"category": "Entertainment", "subcategory": "Streaming"},
            "Prime Video": {"category": "Entertainment", "subcategory": "Streaming"},
            
            # Healthcare
            "Apollo Pharmacy": {"category": "Healthcare", "subcategory": "Pharmacy"},
            "Max Healthcare": {"category": "Healthcare", "subcategory": "Hospital"},
            "Fortis": {"category": "Healthcare", "subcategory": "Hospital"},
            
            # Education
            "Coursera": {"category": "Education", "subcategory": "Online Courses"},
            "Udemy": {"category": "Education", "subcategory": "Online Courses"},
            "Khan Academy": {"category": "Education", "subcategory": "Online Courses"},
        }
        
        self.merchant_patterns = {
            r'(?i)mcdon|mcd\s': "McDonald's",
            r'(?i)starbucks|sbux': "Starbucks", 
            r'(?i)uber|uber\s': "Uber",
            r'(?i)amazon|amzn': "Amazon",
            r'(?i)netflix|nflx': "Netflix",
            r'(?i)airtel|airt': "Airtel",
            r'(?i)jio|rjio': "Jio",
            r'(?i)apollo|apo': "Apollo Pharmacy",
        }
    
    def find_merchant(self, description: str) -> tuple[str, str, str]:
        description_lower = description.lower()
        
        for merchant, info in self.merchants.items():
            if merchant.lower() in description_lower:
                return merchant, info["category"], info["subcategory"]
        
        for pattern, merchant in self.merchant_patterns.items():
            if re.search(pattern, description):
                info = self.merchants.get(merchant, {"category": "Other", "subcategory": "Unknown"})
                return merchant, info["category"], info["subcategory"]
        
        return "Unknown", "Other", "Unknown"

# File Parser
class FileParser:
    def __init__(self):
        self.merchant_db = MerchantDatabase()
    
    def parse_file(self, file_path: str) -> List[Dict]:
        file_ext = Path(file_path).suffix.lower()
        
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        if file_ext == '.pdf':
            return self.parse_pdf(file_content)
        elif file_ext == '.csv':
            return self.parse_csv(file_content)
        elif file_ext in ['.xlsx', '.xls']:
            return self.parse_excel(file_content)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def parse_pdf(self, file_content: bytes) -> List[Dict]:
        transactions = []
        doc = fitz.open(stream=file_content, filetype="pdf")
        all_text = ""
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            if not text.strip():
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                text = pytesseract.image_to_string(img)
            
            all_text += text + "\n"
        
        doc.close()
        return self.extract_transactions_from_text(all_text)
    
    def parse_csv(self, file_content: bytes) -> List[Dict]:
        try:
            df = pd.read_csv(io.BytesIO(file_content))
            return self.extract_transactions_from_dataframe(df)
        except Exception as e:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for delimiter in [',', ';', '\t']:
                    try:
                        df = pd.read_csv(io.BytesIO(file_content), 
                                       encoding=encoding, delimiter=delimiter)
                        return self.extract_transactions_from_dataframe(df)
                    except:
                        continue
            raise ValueError(f"Could not parse CSV: {e}")
    
    def parse_excel(self, file_content: bytes) -> List[Dict]:
        try:
            xl_file = pd.ExcelFile(io.BytesIO(file_content))
            
            for sheet_name in xl_file.sheet_names:
                try:
                    df = pd.read_excel(io.BytesIO(file_content), sheet_name=sheet_name)
                    if len(df) > 0:
                        transactions = self.extract_transactions_from_dataframe(df)
                        if transactions:
                            return transactions
                except:
                    continue
            
            df = pd.read_excel(io.BytesIO(file_content))
            return self.extract_transactions_from_dataframe(df)
        except Exception as e:
            raise ValueError(f"Could not parse Excel: {e}")
    
    def extract_transactions_from_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        transactions = []
        
        column_mappings = {
            'date': ['date', 'transaction_date', 'txn_date', 'dt', 'trans_date'],
            'description': ['description', 'particulars', 'narration', 'details', 'transaction_details'],
            'amount': ['amount', 'amt', 'transaction_amount', 'txn_amt'],
            'credit': ['credit', 'cr', 'credit_amount', 'deposit'],
            'debit': ['debit', 'dr', 'debit_amount', 'withdrawal'],
            'balance': ['balance', 'running_balance', 'available_balance']
        }
        
        df_columns_lower = [col.lower() for col in df.columns]
        identified_columns = {}
        
        for standard_col, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df_columns_lower:
                    original_col_name = df.columns[df_columns_lower.index(possible_name)]
                    identified_columns[standard_col] = original_col_name
                    break
        
        for _, row in df.iterrows():
            try:
                transaction = self.extract_transaction_from_row(row, identified_columns)
                if transaction:
                    transactions.append(transaction)
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue
        
        return transactions
    
    def extract_transaction_from_row(self, row: pd.Series, column_map: Dict) -> Optional[Dict]:
        try:
            date_str = str(row.get(column_map.get('date', ''), '')).strip()
            description = str(row.get(column_map.get('description', ''), '')).strip()
            
            if not date_str or not description or date_str == 'nan':
                return None
            
            try:
                parsed_date = date_parser.parse(date_str, fuzzy=True)
                formatted_date = parsed_date.strftime('%Y-%m-%d')
            except:
                formatted_date = date_str
            
            amount = 0.0
            transaction_type = "Unknown"
            
            if 'credit' in column_map and 'debit' in column_map:
                credit_val = row.get(column_map['credit'], 0)
                debit_val = row.get(column_map['debit'], 0)
                
                if pd.notna(credit_val) and float(credit_val) > 0:
                    amount = float(credit_val)
                    transaction_type = "Credit"
                elif pd.notna(debit_val) and float(debit_val) > 0:
                    amount = float(debit_val)
                    transaction_type = "Debit"
            elif 'amount' in column_map:
                amount_val = row.get(column_map['amount'], 0)
                if pd.notna(amount_val):
                    amount = abs(float(amount_val))
                    transaction_type = "Debit" if float(amount_val) < 0 else "Credit"
            
            merchant_name, category, subcategory = self.merchant_db.find_merchant(description)
            
            return {
                'date': formatted_date,
                'amount': amount,
                'transaction_type': transaction_type,
                'merchant_name': merchant_name,
                'category': category,
                'subcategory': subcategory,
                'account_involved': 'Main Account',
                'location': self.extract_location(description),
                'description': description
            }
        except Exception as e:
            logger.error(f"Error extracting transaction: {e}")
            return None
    
    def extract_transactions_from_text(self, text: str) -> List[Dict]:
        transactions = []
        lines = text.split('\n')
        
        transaction_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(.+?)\s+([\d,]+\.?\d*)\s*(CR|DR)?',
            r'(\d{4}-\d{1,2}-\d{1,2})\s+(.+?)\s+([\d,]+\.?\d*)\s*(CR|DR)?',
            r'(\d{1,2}\s+\w+\s+\d{4})\s+(.+?)\s+([\d,]+\.?\d*)\s*(CR|DR)?'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            for pattern in transaction_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        date_str, description, amount_str, cr_dr = match.groups()
                        
                        try:
                            parsed_date = date_parser.parse(date_str, fuzzy=True)
                            formatted_date = parsed_date.strftime('%Y-%m-%d')
                        except:
                            formatted_date = date_str
                        
                        amount = float(amount_str.replace(',', ''))
                        transaction_type = "Credit" if cr_dr and cr_dr.upper() == "CR" else "Debit"
                        
                        merchant_name, category, subcategory = self.merchant_db.find_merchant(description)
                        
                        transaction = {
                            'date': formatted_date,
                            'amount': amount,
                            'transaction_type': transaction_type,
                            'merchant_name': merchant_name,
                            'category': category,
                            'subcategory': subcategory,
                            'account_involved': 'Main Account',
                            'location': self.extract_location(description),
                            'description': description.strip()
                        }
                        
                        transactions.append(transaction)
                        break
                    except Exception as e:
                        logger.warning(f"Error parsing transaction line: {e}")
                        continue
        
        return transactions
    
    def extract_location(self, description: str) -> Optional[str]:
        location_patterns = [
            r'(?i)\b([A-Z][a-z]+ ?[A-Z][a-z]+)\b',
            r'(?i)\b([A-Z]{2,})\b',
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, description)
            if matches:
                return matches[0] if isinstance(matches[0], str) else matches[0][0]
        
        return None

# Transaction Categorizer
class TransactionCategorizer:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.merchant_db = MerchantDatabase()
    
    def load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False
        )
    
    def categorize_transaction(self, description: str) -> tuple[str, str, str]:
        merchant_name, category, subcategory = self.merchant_db.find_merchant(description)
        
        if merchant_name != "Unknown":
            return merchant_name, category, subcategory
        
        prompt = f"<s>[INST] Categorize this transaction: {description} [/INST]"
        
        try:
            response = self.pipeline(prompt)
            result = response[0]['generated_text'].split('[/INST]')[-1].strip()
            
            if "Category:" in result:
                parts = result.split(",")
                category = parts[0].split("Category:")[-1].strip()
                subcategory = parts[1].split("Subcategory:")[-1].strip() if len(parts) > 1 else "Unknown"
                merchant = parts[2].split("Merchant:")[-1].strip() if len(parts) > 2 else "Unknown"
                return merchant, category, subcategory
            
        except Exception as e:
            logger.error(f"Error in AI categorization: {e}")
        
        return "Unknown", "Other", "Unknown"

# Main Processor
class BankStatementAnalyzer:
    def __init__(self):
        self.file_parser = FileParser()
        self.categorizer = TransactionCategorizer()
        self.categorizer.load_model()
    
    def analyze_file(self, file_path: str):
        print(f"\nAnalyzing file: {file_path}")
        start_time = datetime.now()
        
        try:
            raw_transactions = self.file_parser.parse_file(file_path)
            
            if not raw_transactions:
                print("No transactions found in the file")
                return
            
            print(f"\nFound {len(raw_transactions)} transactions:")
            print("-" * 100)
            
            for i, raw_txn in enumerate(raw_transactions, 1):
                try:
                    if raw_txn['merchant_name'] == "Unknown":
                        merchant, category, subcategory = self.categorizer.categorize_transaction(
                            raw_txn['description']
                        )
                        raw_txn['merchant_name'] = merchant
                        raw_txn['category'] = category
                        raw_txn['subcategory'] = subcategory
                    
                    print(f"Transaction #{i}:")
                    print(f"Date: {raw_txn['date']}")
                    print(f"Amount: {raw_txn['amount']} ({raw_txn['transaction_type']})")
                    print(f"Merchant: {raw_txn['merchant_name']}")
                    print(f"Category: {raw_txn['category']} > {raw_txn['subcategory']}")
                    print(f"Description: {raw_txn['description']}")
                    if raw_txn['location']:
                        print(f"Location: {raw_txn['location']}")
                    print("-" * 50)
                    
                except Exception as e:
                    logger.error(f"Error processing transaction: {e}")
                    continue
            
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"\nAnalysis completed in {processing_time:.2f} seconds")
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            print(f"\nError processing file: {str(e)}")
            print(f"Failed after {processing_time:.2f} seconds")

# Main function
def main():
    print("AI Bank Statement Analyzer")
    print("=" * 50)
    
    analyzer = BankStatementAnalyzer()
    
    while True:
        file_path = input("\nEnter path to bank statement file (PDF/CSV/XLSX) or 'q' to quit: ").strip()
        
        if file_path.lower() == 'q':
            break
        
        if not os.path.exists(file_path):
            print("File not found. Please try again.")
            continue
        
        analyzer.analyze_file(file_path)

if __name__ == "__main__":
    main()