from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.ml_service import ml_service
from app.services.xai_service import xai_service

router = APIRouter()

class EmployeeDataInput(BaseModel):
    no_of_trainings: int = Field(default=1, ge=1)
    Age: int = Field(default=30, ge=18)
    length_of_service: int = Field(default=3, ge=0)
    awards_won: int = Field(default=0, ge=0, le=1)
    avg_training_score: int = Field(default=60, ge=0, le=100)
    Department: str = Field(default="Engineering")
    region: str = Field(default="region_1")
    EducationLevel: str = Field(default="Bachelor's")
    Gender: str = Field(default="Male")
    recruitment_channel: str = Field(default="sourcing")
    DailyRate: int = Field(default=800)
    DistanceFromHome: int = Field(default=5)
    EducationNumerical: int = Field(default=3)
    EnvironmentSatisfaction: int = Field(default=3, ge=1, le=4)
    HourlyRate: int = Field(default=65)
    JobInvolvement: int = Field(default=3, ge=1, le=4)
    JobLevel: int = Field(default=2)
    JobSatisfaction: int = Field(default=3, ge=1, le=4)
    CurrentSalary: int = Field(default=60000)
    NumCompaniesWorked: int = Field(default=2)
    PercentSalaryHike: int = Field(default=12)
    RelationshipSatisfaction: int = Field(default=3, ge=1, le=4)
    Total_Experience_Years: int = Field(default=5)
    WorkLifeBalance: int = Field(default=3, ge=1, le=4)
    YearsInCurrentRole: int = Field(default=2)
    YearsSinceLastPromotion: int = Field(default=1)
    YearsWithCurrManager: int = Field(default=2)
    BusinessTravel: str = Field(default="Travel_Rarely")
    EducationField: str = Field(default="Medical")
    JobRole: str = Field(default="Sales Executive")
    MaritalStatus: str = Field(default="Single")
    OverTime: str = Field(default="No")
    JobTitle: str = Field(default="Software Engineer")
    Country: str = Field(default="UK")
    Race: str = Field(default="White")

@router.post("/predict")
async def process_employee_evaluation(payload: EmployeeDataInput):
    """
    Inferences pipeline endpoint processing corporate operational variables 
    simultaneously across 4 discrete ML systems and generating Gemini XAI summaries.
    """
    try:
        input_data = payload.model_dump()
        
        
        predictions = ml_service.predict_single_employee(input_data)
        
         
        narrative_breakdown = xai_service.generate_employee_narrative(input_data, predictions)
        
        return {
            "status": "success",
            "data": predictions,
            "insights_summary": narrative_breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Processing Engine Error: {str(e)}")