from pydantic import BaseModel, Field, computed_field
from typing import Annotated

class CustomerYearlyMetrics(BaseModel):
    Transactions : Annotated[int, Field(...,gt = 0, description = 'Total number of Transactions by the user')]
    Quantity : Annotated[int, Field(...,gt = 0, description = 'Sum of all Quantities in all transations by the user')]
    Spent : Annotated[float, Field(...,gt = 0, description = 'Sum of all Money spent in all transations by the user')]
    Refunds : Annotated[int, Field(...,ge = 0, description = 'Sum of all Money sent to user in all transations by the user')]
    Returns : Annotated[int, Field(...,ge = 0, description = 'Amount of return transistions')]
    
    @computed_field
    @property
    def Avg_Spent(self)-> float:
        return self.Spent / self.Transactions
    
    @computed_field
    @property
    def Transactions_per_month(self)-> float:
        return self.Transactions / 12
    
    @computed_field
    @property
    def Return_Rate(self)-> float:
        if self.Returns:
            return self.Transactions / self.Returns 
        return 0
    
