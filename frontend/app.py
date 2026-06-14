import streamlit as st
import requests


st.set_page_config(
    page_title="KARMA-AI | Core Talent Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header { font-size:2.2rem !important; color: #1E88E5; font-weight: 700; margin-bottom: 0.2rem; }
    .main-subtitle { font-size:1.2rem !important; color: #555555; font-weight: 400; margin-top: 0rem; margin-bottom: 0.8rem; font-style: italic; }
    .section-subheader { font-size:1.3rem !important; fwith st.sidebar.form(key="employee_form"):ont-weight: 600; margin-top: 1rem; margin-bottom: 1rem; }
    .metric-card { background-color: #f8f9fa; border-radius: 8px; padding: 15px; border-left: 5px solid #1E88E5; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header"> KARMA-AI: Knowledge-driven Analytics for Retention, Merit & Advancement</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Predictive Workforce Intelligence & AI-Driven Retention Analytics</p>', unsafe_allow_html=True)
st.markdown("---")


st.sidebar.markdown("### 📋 Employee Evaluation Profile")
st.sidebar.markdown("Input organizational metrics below to recalculate risks.")

with st.sidebar.form(key="employee_form"):
    st.markdown("#### **Primary Demographics & Role**")
    
    job_title = st.selectbox("Job Title / Category", [
        "Software Engineer", "Data Scientist", "Sales Executive", "Manager", "Research Scientist",
        "Product Manager", "HR Specialist", "Financial Analyst", "Operations Coordinator", "Other / Executive Specialist"
    ])
    
    department = st.selectbox("Department", [
        "Engineering", "Sales", "Research & Development", "Human Resources", "Finance", 
        "Operations", "Marketing", "Legal", "Customer Success"
    ])
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary / Prefer Not to Say"])
    
    st.markdown("---")
    st.markdown("#### **Performance & Tenure**")
    total_exp = st.slider("Total Experience (Years)", 0, 50, 5) 
    tenure = st.slider("Length of Service / Tenure at Company (Years)", 0, 40, 3) 
    trainings = st.slider("Number of Trainings Completed Last Year", 1, 10, 1)
    training_score = st.slider("Average Training Assessment Score", 0, 100, 60)
    awards = st.checkbox("Won Company Awards Last Year?")

    st.markdown("---")
    st.markdown("#### **Compensation & Satisfaction**")
    current_salary = st.number_input("Current Annual Salary ($ equivalent)", min_value=5000, max_value=1000000, value=60000, step=5000)
    
    
    env_satisfaction = st.slider("Work Environment Satisfaction (1=Low, 4=Exceptional)", 1, 4, 3)
    job_satisfaction = st.slider("Job Roles/Tasks Satisfaction (1=Low, 4=Exceptional)", 1, 4, 3)
    work_life = st.slider("Work-Life Balance Rating (1=Poor, 4=Excellent)", 1, 4, 3)
    overtime = st.selectbox("Regularly Logs Overtime?", ["No", "Yes"])

    st.markdown("---")
    st.markdown("#### **Global Demographic Nodes**")
    
    country = st.selectbox("Country / Regional Base", [
        "India", "US", "UK", "Canada", "Australia", "Europe (Other)", "Asia (Other)", "Latin America", "Africa Node"
    ])
    
    race = st.selectbox("Demographic / Ethnic Group", [
        "Asian (including Indian/East Asian)", "White", "Black / African Descent", "Hispanic / Latino", "Middle Eastern", "Indigenous / Other"
    ])

    submit_button = st.form_submit_button(label="⚡ Execute Strategic Inference")


if submit_button:
    
    payload = {
        "no_of_trainings": int(trainings),
        "Age": int(age),
        "length_of_service": int(tenure),
        "awards_won": 1 if awards else 0,
        "avg_training_score": int(training_score),
        "Department": str(department),
        "region": "region_1", 
        "EducationLevel": "Bachelor's", 
        "Gender": str(gender),
        "recruitment_channel": "sourcing",
        "DailyRate": 800,
        "DistanceFromHome": 5,
        "EducationNumerical": 3,
        "EnvironmentSatisfaction": int(env_satisfaction),
        "HourlyRate": 65,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobSatisfaction": int(job_satisfaction),
        "CurrentSalary": int(current_salary),
        "NumCompaniesWorked": 2,
        "PercentSalaryHike": 12,
        "RelationshipSatisfaction": 3,
        "Total_Experience_Years": int(total_exp),
        "WorkLifeBalance": int(work_life),
        "YearsInCurrentRole": int(max(0, tenure - 1)),
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": int(max(0, tenure - 1)),
        "BusinessTravel": "Travel_Rarely",
        "EducationField": "Medical",
        "JobRole": "Sales Executive",
        "MaritalStatus": "Single",
        "OverTime": str(overtime),
        "JobTitle": str(job_title),
        "Country": str(country),
        "Race": str(race)
    }

    BACKEND_URL= "https://asminsinha2005-karma-ai-backend.hf.space/api/v1/predict"
    
    with st.spinner("🧠 Orchestrating predictive matrices and XAI diagnostics..."):
        try:
            response = requests.post(BACKEND_URL, json=payload)
            if response.status_code == 200:
                res_json = response.json()
                metrics = res_json["data"]
                narrative = res_json["insights_summary"]

                
                st.markdown('<p class="section-subheader">📊 Calculated Core Multi-Model Indices</p>', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(label="📈 Promotion Probability", value=f"{metrics['promotion_probability']}%")
                with col2:
                    st.metric(label="🎯 Performance Forecast", value=f"{metrics['performance_forecast_rating']} / 5.0")
                with col3:
                    
                    attr_risk = metrics['attrition_risk_score']
                    st.metric(label="⚠️ Attrition Risk Index", value=f"{attr_risk}%")
                with col4:
                    st.metric(label="💵 Predicted Market Value", value=f"${metrics['predicted_market_salary']:,.2f}")

                st.markdown("---")
                
                
                st.markdown('<p class="section-subheader">🔮 KARMA-AI Generative XAI Executive Breakdown</p>', unsafe_allow_html=True)
                st.info("The insights below combine real machine learning model outputs with generative natural language explanations.")
                st.markdown(narrative)

            else:
                st.error(f" Core Gateway Error: Received Status Code {response.status_code}")
                st.write(response.text)
        except Exception as e:
            st.error(f" Communication Failure: Unable to connect to FastAPI Server at {BACKEND_URL}")
            st.write(f"Ensure your FastAPI server is currently running in your second terminal pane via: `uvicorn app.main:app --reload`")
else:
    
    st.info("👈 Complete the parameters in the Employee Evaluation Profile sidebar and click 'Execute Strategic Inference' to compute live diagnostic analysis dashboards.")