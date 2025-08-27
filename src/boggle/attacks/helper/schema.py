from pydantic import BaseModel


class UnharmfulData(BaseModel):
    rephrased_text: str
    
class AttackData(BaseModel):
    generated_text: str
    
class RefusalData(BaseModel):
    refused: bool
    
class HazardScoreData(BaseModel):
    hazard_score: float
    
class SemanticScoreData(BaseModel):
    semantic_score: float
    

    
    