from typing import List, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re

class Transaction(BaseModel):
    """Individual transaction record with validation."""
    from_field: str = Field(..., alias="from", description="Source of the transaction")
    to: str = Field(..., description="Destination/recipient of the transaction")
    credit_account: str = Field(..., description="Account being credited")
    debit_account: str = Field(..., description="Account being debited")
    expense_type: str = Field(..., description="Primary expense category")
    subcategory: str = Field(..., description="Expense subcategory")
    debit_amount: float = Field(0.0, ge=0, description="Amount debited")
    credit_amount: float = Field(0.0, ge=0, description="Amount credited")
    transaction_date: str = Field(..., description="Transaction date in YYYY-MM-DD format")
    
    @validator('transaction_date')
    def validate_date_format(cls, v):
        """Ensure date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')
    
    @validator('debit_amount', 'credit_amount', pre=True)
    def clean_amounts(cls, v):
        """Clean and validate monetary amounts."""
        if isinstance(v, str):
            # Remove currency symbols, commas, and whitespace
            cleaned = re.sub(r'[^\d.-]', '', v.strip())
            if not cleaned or cleaned == '-':
                return 0.0
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        return float(v) if v is not None else 0.0
    
    class Config:
        populate_by_name = True
        json_encoders = {
            float: lambda v: round(v, 2)
        }

class TransactionList(BaseModel):
    """List of validated transactions."""
    transactions: List[Transaction]
    total_count: int = Field(..., description="Total number of transactions")
    processing_metadata: Optional[dict] = Field(None, description="Processing information")
    
    @validator('total_count')
    def validate_count(cls, v, values):
        """Ensure count matches actual transaction list length."""
        if 'transactions' in values:
            actual_count = len(values['transactions'])
            if v != actual_count:
                return actual_count
        return v