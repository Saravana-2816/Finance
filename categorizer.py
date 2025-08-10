import re
import json
import logging
import requests
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
from rapidfuzz import fuzz
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class TransactionCategorizer:
    """Categorizes transactions using LLM and embeddings."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", gemini_api_key: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ollama_url = ollama_url
        
        # Initialize Gemini for embeddings
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.use_embeddings = True
        else:
            self.logger.warning("No Gemini API key provided. Embeddings will be disabled.")
            self.use_embeddings = False
        
        # Vendor mapping dictionary
        self.vendor_mappings = {
            # Food & Dining
            'MCD': 'McDonald\'s',
            'MCDONALDS': 'McDonald\'s',
            'KFC': 'KFC',
            'STARBUCKS': 'Starbucks',
            'SUBWAY': 'Subway',
            'PIZZA HUT': 'Pizza Hut',
            'DOMINOS': 'Domino\'s Pizza',
            'BURGER KING': 'Burger King',
            'TACO BELL': 'Taco Bell',
            'CHIPOTLE': 'Chipotle',
            
            # Retail
            'WALMART': 'Walmart',
            'TARGET': 'Target',
            'AMAZON': 'Amazon',
            'COSTCO': 'Costco',
            'HOME DEPOT': 'Home Depot',
            'LOWES': 'Lowe\'s',
            'BEST BUY': 'Best Buy',
            'MACYS': 'Macy\'s',
            
            # Gas Stations
            'SHELL': 'Shell',
            'CHEVRON': 'Chevron',
            'EXXON': 'ExxonMobil',
            'BP': 'BP',
            'TEXACO': 'Texaco',
            'MOBIL': 'Mobil',
            
            # Banks & Financial
            'BANK OF AMERICA': 'Bank of America',
            'CHASE': 'JPMorgan Chase',
            'WELLS FARGO': 'Wells Fargo',
            'CITI': 'Citibank',
            'ATM FEE': 'ATM Fee',
            'OVERDRAFT': 'Overdraft Fee',
            
            # Utilities
            'ELECTRIC': 'Electric Company',
            'GAS COMPANY': 'Gas Company',
            'WATER DEPT': 'Water Department',
            'INTERNET': 'Internet Provider',
            'PHONE': 'Phone Company',
        }
        
        # Category definitions for embedding similarity
        self.category_definitions = {
            'Food & Dining': [
                'restaurant meals and dining out',
                'fast food purchases',
                'coffee shops and cafes',
                'food delivery services',
                'grocery stores and supermarkets'
            ],
            'Transportation': [
                'gas stations and fuel purchases',
                'public transportation fares',
                'ride sharing services like uber and lyft',
                'parking fees and tolls',
                'vehicle maintenance and repairs'
            ],
            'Shopping': [
                'retail stores and shopping',
                'online purchases and e-commerce',
                'clothing and apparel stores',
                'electronics and gadget purchases',
                'home improvement and hardware stores'
            ],
            'Utilities': [
                'electricity and power bills',
                'natural gas utilities',
                'water and sewer services',
                'internet and phone services',
                'cable and streaming services'
            ],
            'Banking & Finance': [
                'bank fees and charges',
                'atm withdrawal fees',
                'interest payments and charges',
                'money transfers and wires',
                'investment and trading fees'
            ],
            'Healthcare': [
                'medical doctor visits',
                'pharmacy and prescription medications',
                'dental and orthodontic services',
                'vision and eye care services',
                'hospital and clinic visits'
            ],
            'Entertainment': [
                'movie theaters and cinemas',
                'streaming services like netflix',
                'gaming and video games',
                'sports and recreational activities',
                'concerts and live events'
            ],
            'Other': [
                'miscellaneous and unclassified expenses',
                'unknown or unclear transactions',
                'other general purchases'
            ]
        }
        
        # Pre-compute category embeddings if Gemini is available
        self.category_embeddings = {}
        if self.use_embeddings:
            self._precompute_category_embeddings()
    
    def _precompute_category_embeddings(self):
        """Pre-compute embeddings for all category definitions."""
        self.logger.info("Pre-computing category embeddings...")
        
        for category, definitions in self.category_definitions.items():
            embeddings = []
            for definition in definitions:
                try:
                    embedding = self._get_gemini_embedding(definition)
                    if embedding:
                        embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Failed to get embedding for {definition}: {e}")
            
            if embeddings:
                # Average the embeddings for this category
                self.category_embeddings[category] = np.mean(embeddings, axis=0)
        
        self.logger.info(f"Pre-computed embeddings for {len(self.category_embeddings)} categories")
    
    def _get_gemini_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Gemini API."""
        if not self.use_embeddings:
            return None
            
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="semantic_similarity"
            )
            return np.array(result['embedding'])
        except Exception as e:
            self.logger.error(f"Gemini embedding error: {e}")
            return None
    
    def _query_mistral(self, prompt: str) -> str:
        """Query Mistral 7B via Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "mistral:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 200
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                self.logger.error(f"Ollama API error: {response.status_code}")
                return ""
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error querying Mistral via Ollama: {e}")
            return ""
    
    def categorize_transaction(self, transaction: Dict) -> Tuple[str, str, str, str]:
        """
        Categorize a transaction using Mistral LLM and Gemini embeddings.
        
        Args:
            transaction: Transaction dictionary with raw_description
            
        Returns:
            Tuple of (from_account, to_account, expense_type, subcategory)
        """
        description = transaction.get('raw_description', '').strip()
        
        # Extract vendor name using mappings and LLM
        vendor_name = self._extract_vendor_name(description)
        
        # Categorize using hybrid approach: embeddings + LLM
        expense_type, subcategory = self._categorize_hybrid(description, vendor_name)
        
        # Determine from/to accounts
        from_account, to_account = self._determine_accounts(transaction, vendor_name)
        
        return from_account, to_account, expense_type, subcategory
    
    def _extract_vendor_name(self, description: str) -> str:
        """Extract vendor name using mappings and Mistral LLM."""
        # First check direct mappings
        description_upper = description.upper()
        
        for key, vendor in self.vendor_mappings.items():
            if key in description_upper:
                return vendor
        
        # Use fuzzy matching for partial matches
        best_match = ""
        best_score = 0
        
        for key, vendor in self.vendor_mappings.items():
            score = fuzz.partial_ratio(key.lower(), description.lower())
            if score > 80 and score > best_score:
                best_match = vendor
                best_score = score
        
        if best_match:
            return best_match
        
        # Use Mistral to extract vendor name
        vendor_name = self._extract_vendor_with_llm(description)
        if vendor_name and vendor_name != "Unknown":
            return vendor_name
        
        # Fallback: extract potential company name
        words = description.split()
        for i, word in enumerate(words[:3]):
            if word.isupper() and len(word) > 2:
                potential_name = ' '.join(words[i:i+2]).title()
                return potential_name
        
        return "Unknown Vendor"
    
    def _extract_vendor_with_llm(self, description: str) -> str:
        """Use Mistral to extract vendor/merchant name."""
        prompt = f"""Extract the merchant or vendor name from this transaction description. Return only the business name, nothing else.

Transaction: {description}

Merchant name:"""
        
        response = self._query_mistral(prompt)
        
        if response and len(response) < 50:  # Reasonable business name length
            # Clean the response
            cleaned = response.split('\n')[0].strip()
            if cleaned and not cleaned.lower() in ['unknown', 'n/a', 'none', 'not available']:
                return cleaned.title()
        
        return "Unknown"
    
    def _categorize_hybrid(self, description: str, vendor_name: str) -> Tuple[str, str]:
        """Categorize using embeddings similarity + LLM reasoning."""
        # Method 1: Embedding-based similarity
        embedding_category = self._categorize_with_embeddings(description)
        
        # Method 2: LLM-based categorization
        llm_category, llm_subcategory = self._categorize_with_llm(description, vendor_name)
        
        # Combine results (prefer LLM if both methods agree on main category)
        if embedding_category and llm_category:
            if embedding_category == llm_category:
                return llm_category, llm_subcategory
            else:
                # Use LLM result but log the disagreement
                self.logger.debug(f"Category disagreement - Embedding: {embedding_category}, LLM: {llm_category}")
                return llm_category, llm_subcategory
        
        # Fallback to whichever method succeeded
        if llm_category:
            return llm_category, llm_subcategory
        elif embedding_category:
            return embedding_category, "General"
        else:
            return "Other", "Miscellaneous"
    
    def _categorize_with_embeddings(self, description: str) -> Optional[str]:
        """Categorize using Gemini embeddings similarity."""
        if not self.use_embeddings or not self.category_embeddings:
            return None
        
        try:
            # Get embedding for the transaction description
            desc_embedding = self._get_gemini_embedding(description)
            if desc_embedding is None:
                return None
            
            # Calculate similarities with each category
            best_category = None
            best_similarity = -1
            
            for category, cat_embedding in self.category_embeddings.items():
                similarity = cosine_similarity(
                    desc_embedding.reshape(1, -1), 
                    cat_embedding.reshape(1, -1)
                )[0][0]
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
            
            # Only return if similarity is above threshold
            if best_similarity > 0.7:  # Adjust threshold as needed
                self.logger.debug(f"Embedding categorization: {best_category} (similarity: {best_similarity:.3f})")
                return best_category
            
        except Exception as e:
            self.logger.error(f"Embedding categorization error: {e}")
        
        return None
    
    def _categorize_with_llm(self, description: str, vendor_name: str) -> Tuple[str, str]:
        """Categorize using Mistral LLM."""
        categories_list = ", ".join(self.category_definitions.keys())
        
        prompt = f"""Categorize this transaction into the most appropriate category and subcategory.

Transaction: {description}
Vendor: {vendor_name}

Available categories: {categories_list}

Analyze the transaction and respond in this exact JSON format:
{{"category": "category_name", "subcategory": "subcategory_name"}}

Only use categories from the provided list. Be specific with subcategories (e.g., "Fast Food", "Gas & Fuel", "Online Shopping", etc.).

Response:"""
        
        response = self._query_mistral(prompt)
        
        try:
            # Try to parse JSON response
            if '{' in response and '}' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
                result = json.loads(json_str)
                
                category = result.get('category', 'Other')
                subcategory = result.get('subcategory', 'Miscellaneous')
                
                # Validate category exists
                if category in self.category_definitions:
                    return category, subcategory
        
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to parse LLM categorization response: {e}")
        
        # Fallback: rule-based categorization
        return self._categorize_fallback(description, vendor_name)
    
    def _categorize_fallback(self, description: str, vendor_name: str) -> Tuple[str, str]:
        """Fallback rule-based categorization."""
        desc_lower = description.lower()
        vendor_lower = vendor_name.lower()
        
        # Simple keyword-based rules
        if any(word in desc_lower for word in ['mcdonalds', 'kfc', 'burger', 'pizza', 'restaurant', 'food']):
            return 'Food & Dining', 'Fast Food'
        elif any(word in desc_lower for word in ['gas', 'fuel', 'shell', 'chevron', 'exxon']):
            return 'Transportation', 'Gas & Fuel'
        elif any(word in desc_lower for word in ['walmart', 'target', 'amazon', 'store']):
            return 'Shopping', 'General Merchandise'
        elif any(word in desc_lower for word in ['electric', 'gas company', 'water', 'internet']):
            return 'Utilities', 'General Utilities'
        elif any(word in desc_lower for word in ['fee', 'atm', 'bank', 'overdraft']):
            return 'Banking & Finance', 'Bank Fees'
        else:
            return 'Other', 'Miscellaneous'
    
    def _determine_accounts(self, transaction: Dict, vendor_name: str) -> Tuple[str, str]:
        """Determine from and to accounts based on transaction type."""
        debit_amount = transaction.get('debit_amount', 0)
        credit_amount = transaction.get('credit_amount', 0)
        
        if debit_amount > 0:
            # Money going out - from user's account to vendor
            from_account = "My Account"
            to_account = vendor_name
        elif credit_amount > 0:
            # Money coming in - from vendor/source to user's account
            from_account = vendor_name
            to_account = "My Account"
        else:
            # Fallback
            from_account = "Unknown"
            to_account = "Unknown"
        
        return from_account, to_account