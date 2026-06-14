import os
from google import genai
from google.genai import types
from app.core.config import settings

class XAIService:
    def __init__(self):
        print("🔮 Initializing Generative Explainable AI Engine...")
        
        self.api_key = settings.GEMINI_API_KEY
        self._client = None

    @property
    def client(self):
        """Lazy load the GenAI client to handle runtime environmental fallback."""
        if self._client is None:
            if not self.api_key:
                
                env_key = os.getenv("GEMINI_API_KEY")
                if not env_key:
                    raise ValueError(" CRITICAL: Gemini API Key is missing. Update backend/.env file.")
                self._client = genai.Client(api_key=env_key)
            else:
                self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate_employee_narrative(self, input_features: dict, predictions: dict) -> str:
        """
        Takes raw employee traits and calculated machine learning outputs, constructs
        a corporate analysis matrix, and extracts a high-utility text overview via Gemini.
        """
        try:
            
            prompt = f"""
            You are KARMA-AI, an expert enterprise industrial psychologist and executive talent analytics system.
            Analyze the following machine learning predictive metrics alongside the employee's operational profile and generate a sharp, professional executive summary.

            ### EMPLOYEE RECORD SUMMARY:
            - Age: {input_features.get('Age', 30)}
            - Title/Role: {input_features.get('JobTitle', 'Software Engineer')} (Level {input_features.get('JobLevel', 2)})
            - Department: {input_features.get('Department', 'Engineering')}
            - Current Tenure: {input_features.get('length_of_service', 3)} Years (Total Experience: {input_features.get('Total_Experience_Years', 5)} Years)
            - Logged Overtime: {input_features.get('OverTime', 'No')}
            - Current Salary: ${input_features.get('CurrentSalary', 60000)}/year

            ### CALCULATED ML CORE PREDICTIONS:
            1. Promotion Probability: {predictions.get('promotion_probability')}%
            2. Predicted Performance Rating: {predictions.get('performance_forecast_rating')}/5.00
            3. Attrition/Turnover Risk: {predictions.get('attrition_risk_score')}%
            4. Fair Market Value Estimation: ${predictions.get('predicted_market_salary')}/year

            ### STRUCTURAL REQUIREMENTS FOR YOUR OUTPUT:
            - Executive Diagnosis: Break down why the attrition score or promotion risk sits at this level based on their inputs (e.g., compensation gaps vs market valuation, overtime burn, job metrics).
            - Retainability Strategy: Provide 2 highly tailored, data-driven action items for management to keep or optimize this employee.
            - Keep your tone concise, strictly corporate, objective, and authoritative. Do not use generic filler prose. Go straight to the bullet points.
            - Ensure your analysis is fully self-contained and finishes all thoughts completely. Do not cut off mid-sentence.
            """

            response = self.client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1200
                )
            )
            return response.text
        except Exception as e:
            return f"⚠️ XAI Generation Warning: Analytics synthesis bypass active. Reason: {str(e)}"

# Instantiate singleton
xai_service = XAIService()