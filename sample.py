from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List
import httpx

# Your target JSON schema
class Transaction(BaseModel):
    transaction_id: str
    sender: str
    receiver: str
    sender_account: str
    receiver_account: str
    amount_credited: float
    amount_debited: float
    expense_category: str   # e.g., "Food", "Travel", "Shopping"
    specific_vendor: str    # e.g., "McDonald's", "AirAsia"

class BankStatementSummary(BaseModel):
    transactions: List[Transaction]

# Initialize client
client = genai.Client(api_key="AIzaSyBAkH5ehQ2aUGDHDSMVaRyNgzomxDwFdyU")

# Fetch PDF
doc_url = r"D:\Desktop\Finance\Finance\Bank-Statement-Template-1-TemplateLab.pdf"
with open(doc_url, "rb") as f:
    doc_data = f.read()
# Craft the **perfect prompt**
prompt = """
You are an expert global financial statement analyzer with deep knowledge of worldwide businesses, brands, services, and merchant categories, including major corporations, small local vendors, and niche businesses from any country or region.

You will be given a bank statement in PDF format.  
Your task is to extract for **each transaction** the following fields, ensuring that none of them are left as “Unknown”, “N/A”, blank, or null. If you cannot find the exact value, intelligently infer it based on the available transaction details:

- Transaction ID → If no explicit ID is found, generate a unique ID starting from value 1 and go on with increassing order (for eg:0001).
- Sender name → If missing, infer from sender account description or statement header.
- Receiver name → If missing, infer from vendor name or transaction description.
- Sender account number → If missing, infer using placeholder format with available data (e.g., "Account of <Sender Name>").
- Receiver account number → If missing, infer using placeholder format with available data (e.g., "Account of <Receiver Name>").
- Amount credited → Always return a numeric value; if unclear, infer from transaction context (credit vs debit).
- Amount debited → Always return a numeric value; if unclear, infer from transaction context (credit vs debit).
- Expense Category → Choose the best possible category dynamically based on the transaction. Do not limit to a fixed list; examples include Food, Travel, Utilities, Healthcare, Education, Rent, Insurance, Groceries, Investments, Subscriptions, Entertainment, Transportation, Taxes, Charity, etc.
- Expense Subcategory → Choose a more specific grouping under the category, based on transaction patterns (e.g., Food → Restaurant, Cafe, Bakery; Travel → Airlines, Taxi, Hotel). Always provide the most contextually accurate subcategory possible.
- Specific Vendor → The exact business name as seen in the transaction; if not explicitly listed, infer from the description, category, or known patterns. Use the raw merchant text if available.

Rules:
1. No field should ever return “Unknown”, “N/A”, null, or blank. Always provide the most reasonable inferred value using the available context.
2. Handle vendors from any country, including micro-local shops, hyper-local brands, and region-specific service providers.
3. For account numbers that are masked or missing, still produce a meaningful placeholder (e.g., "Account of McDonald's").
4. If vendor is not explicit but clues exist (keywords, location names, transaction type), infer the vendor name as precisely as possible.
5. Always ensure the output is **strictly in JSON format** matching the provided schema, without any commentary.

"""

# Call Gemini with PDF + prompt + JSON schema enforcement
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[
        types.Part.from_bytes(data=doc_data, mime_type='application/pdf'),
        prompt
    ],
    config={
        "response_mime_type": "application/json",
        "response_schema": BankStatementSummary
    },
)

# Raw JSON string
print(response.text)

# Parsed Python object
summary: BankStatementSummary = response.parsed
print(summary)
