import os
import joblib
import pandas as pd
import numpy as np
from app.core.config import settings

class MLInferenceService:
    def __init__(self):
        print(" Initializing Core ML Inference Service Engine...")
        self.load_models()

    def load_models(self):
        try:
            
            self.promo_perf_preprocessor = joblib.load(os.path.join(settings.MODELS_DIR, "promo_perf_preprocessor.joblib"))
            self.attrition_preprocessor = joblib.load(os.path.join(settings.MODELS_DIR, "attrition_preprocessor.joblib"))
            self.salary_preprocessor = joblib.load(os.path.join(settings.MODELS_DIR, "salary_preprocessor.joblib"))

           
            self.model_promotion = joblib.load(os.path.join(settings.MODELS_DIR, "model_promotion.joblib"))
            self.model_attrition = joblib.load(os.path.join(settings.MODELS_DIR, "model_attrition.joblib"))
            self.model_performance = joblib.load(os.path.join(settings.MODELS_DIR, "model_performance.joblib"))
            self.model_salary = joblib.load(os.path.join(settings.MODELS_DIR, "model_salary.joblib"))
            
            print(" Success: All 4 models loaded cleanly into operational memory space.")
        except Exception as e:
            print(f" CRITICAL ERROR loading ML models: {str(e)}")
            raise e
        
    def predict_single_employee(self, raw_data: dict) -> dict:
        """
        Accepts raw dictionary features, applies translation layers for non-native 
        global inputs to maintain ML alignment, and evaluates performance.
        """
        
        valid_departments = ['Engineering', 'Sales', 'Research & Development', 'Human Resources', 'Finance']
        valid_titles = ['Software Engineer', 'Data Scientist', 'Sales Executive', 'Manager', 'Research Scientist']
        valid_countries = ['US', 'UK', 'Canada', 'Australia']
        valid_races = ['White', 'Asian', 'Black', 'Hispanic']

        ml_dept = raw_data.get('Department', 'Engineering')
        if ml_dept not in valid_departments:
            ml_dept = 'Engineering' 

        ml_title = raw_data.get('JobTitle', 'Software Engineer')
        if ml_title not in valid_titles:
            ml_title = 'Software Engineer'

        ml_country = raw_data.get('Country', 'UK')
        if ml_country not in valid_countries:
            ml_country = 'UK' 

        ml_race = raw_data.get('Race', 'White')
        if "Asian" in str(raw_data.get('Race', '')):
            ml_race = 'Asian'
        elif ml_race not in valid_races:
            ml_race = 'White'
            
        ml_gender = str(raw_data.get('Gender', 'Male'))
        if ml_gender not in ['Male', 'Female']:
            ml_gender = 'Male'


        promo_features = pd.DataFrame({
            'no_of_trainings': [int(raw_data.get('no_of_trainings', 1))],
            'age': [int(raw_data.get('Age', 30))],
            'length_of_service': [int(raw_data.get('length_of_service', 3))],
            'awards_won': [int(raw_data.get('awards_won', 0))],
            'avg_training_score': [int(raw_data.get('avg_training_score', 60))],
            'department': [ml_dept],
            'region': [raw_data.get('region', 'region_1')],
            'education': [raw_data.get('EducationLevel', "Master's & above")],
            'gender': [ml_gender.lower()[0]], 
            'recruitment_channel': [raw_data.get('recruitment_channel', 'sourcing')]
        })
        promo_transformed = self.promo_perf_preprocessor.transform(promo_features)
        
        promo_prob = float(self.model_promotion.predict_proba(promo_transformed)[0][1])
        perf_pred = float(self.model_performance.predict(promo_transformed)[0])

        
        attr_features = pd.DataFrame({
            'Age': [int(raw_data.get('Age', 30))],
            'DailyRate': [int(raw_data.get('DailyRate', 800))],
            'DistanceFromHome': [int(raw_data.get('DistanceFromHome', 5))],
            'Education': [int(raw_data.get('EducationNumerical', 3))],
            'EnvironmentSatisfaction': [int(raw_data.get('EnvironmentSatisfaction', 3))],
            'HourlyRate': [int(raw_data.get('HourlyRate', 65))],
            'JobInvolvement': [int(raw_data.get('JobInvolvement', 3))],
            'JobLevel': [int(raw_data.get('JobLevel', 2))],
            'JobSatisfaction': [int(raw_data.get('JobSatisfaction', 3))],
            'MonthlyIncome': [int(raw_data.get('CurrentSalary', 60000) / 12)], 
            'NumCompaniesWorked': [int(raw_data.get('NumCompaniesWorked', 2))],
            'PercentSalaryHike': [int(raw_data.get('PercentSalaryHike', 12))],
            'PerformanceRating': [int(np.clip(round(perf_pred), 3, 4))], 
            'RelationshipSatisfaction': [int(raw_data.get('RelationshipSatisfaction', 3))],
            'TotalWorkingYears': [int(raw_data.get('Total_Experience_Years', 5))],
            'TrainingTimesLastYear': [int(raw_data.get('no_of_trainings', 1))],
            'WorkLifeBalance': [int(raw_data.get('WorkLifeBalance', 3))],
            'YearsAtCompany': [int(raw_data.get('length_of_service', 3))],
            'YearsInCurrentRole': [int(raw_data.get('YearsInCurrentRole', 2))],
            'YearsSinceLastPromotion': [int(raw_data.get('YearsSinceLastPromotion', 1))],
            'YearsWithCurrManager': [int(raw_data.get('YearsWithCurrManager', 2))],
            'BusinessTravel': [raw_data.get('BusinessTravel', 'Travel_Rarely')],
            'Department': [ml_dept],
            'EducationField': [raw_data.get('EducationField', 'Medical')],
            'Gender': [ml_gender if ml_gender in ['Male', 'Female'] else 'Male'],
            'JobRole': [raw_data.get('JobRole', 'Sales Executive')],
            'MaritalStatus': [raw_data.get('MaritalStatus', 'Single')],
            'OverTime': [raw_data.get('OverTime', 'No')]
        })
        attr_transformed = self.attrition_preprocessor.transform(attr_features)
        attrition_prob = float(self.model_attrition.predict_proba(attr_transformed)[0][1])

       
        salary_features = pd.DataFrame({
            'Age': [int(raw_data.get('Age', 30))],
            'Years of Experience': [float(raw_data.get('Total_Experience_Years', 5))],
            'Gender': [ml_gender if ml_gender in ['Male', 'Female'] else 'Male'],
            'Education Level': [raw_data.get('EducationLevel', "Bachelor's")],
            'Job Title': [ml_title],
            'Country': [ml_country],
            'Race': [ml_race]
        })
        salary_transformed = self.salary_preprocessor.transform(salary_features)
        predicted_market_salary = float(self.model_salary.predict(salary_transformed)[0])

        return {
            "promotion_probability": round(promo_prob * 100, 2),
            "performance_forecast_rating": round(np.clip(perf_pred, 1.0, 5.0), 2),
            "attrition_risk_score": round(attrition_prob * 100, 2),
            "predicted_market_salary": round(predicted_market_salary, 2)
        }


ml_service = MLInferenceService()