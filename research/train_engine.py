import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

np.random.seed(42)

DATASET_BASE = "datasets"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

ATTRITION_PATH = os.path.join(DATASET_BASE, "dataset1_attrition", "ibm_attrition.csv")
PROMOTION_PATH = os.path.join(DATASET_BASE, "dataset2_promotion", "employee_promotion.csv")
SALARY_PATH = os.path.join(DATASET_BASE, "dataset3_salary", "employee_salary.csv")

print(" Starting KARMA-AI Core Model Training Engine...\n" + "="*60)


print(" Processing Engine 1 (Promotion) & Engine 3 (Performance Rating)...")
df_promo = pd.read_csv(PROMOTION_PATH)


df_promo.columns = df_promo.columns.str.strip()

print(" Dataset 2 Columns verified. Dropping KPI dependency safely.")


df_promo = df_promo.dropna(subset=['previous_year_rating'])


promo_num_features = ['no_of_trainings', 'age', 'length_of_service', 'awards_won', 'avg_training_score']
promo_cat_features = ['department', 'region', 'education', 'gender', 'recruitment_channel']

X_p = df_promo[promo_num_features + promo_cat_features]
y_promo = df_promo['is_promoted']
y_perf = df_promo['previous_year_rating']


promo_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), promo_num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), promo_cat_features)
])


X_p_processed = promo_preprocessor.fit_transform(X_p)
X_train_pr, X_test_pr, y_train_pr, y_test_pr = train_test_split(X_p_processed, y_promo, test_size=0.2, random_state=42)
X_train_pf, X_test_pf, y_train_pf, y_test_pf = train_test_split(X_p_processed, y_perf, test_size=0.2, random_state=42)


print(" Training Model 1: XGBoost Promotion Classifier...")
model_promo = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_promo.fit(X_train_pr, y_train_pr)
print(f"   -> Promotion F1-Score: {f1_score(y_test_pr, model_promo.predict(X_test_pr)):.4f}")


print(" Training Model 3: XGBoost Performance Regressor...")
model_perf = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model_perf.fit(X_train_pf, y_train_pf)
print(f"   -> Performance R² Score: {r2_score(y_test_pf, model_perf.predict(X_test_pf)):.4f}")


joblib.dump(promo_preprocessor, os.path.join(MODELS_DIR, "promo_perf_preprocessor.joblib"))
joblib.dump(model_promo, os.path.join(MODELS_DIR, "model_promotion.joblib"))
joblib.dump(model_perf, os.path.join(MODELS_DIR, "model_performance.joblib"))



print("\n Processing Engine 2 (Attrition Risk)...")
df_attr = pd.read_csv(ATTRITION_PATH)

df_attr['Target_Attrition'] = df_attr['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

attr_num_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
attr_cat_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']

X_a = df_attr[attr_num_features + attr_cat_features]
y_attr = df_attr['Target_Attrition']

attr_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('scaler', StandardScaler())]), attr_num_features),
    ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), attr_cat_features)
])

X_a_processed = attr_preprocessor.fit_transform(X_a)
X_train_at, X_test_at, y_train_at, y_test_at = train_test_split(X_a_processed, y_attr, test_size=0.2, random_state=42, stratify=y_attr)


print(" Training Model 2: Random Forest Attrition Classifier...")
model_attr = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)
model_attr.fit(X_train_at, y_train_at)
print(f"   -> Attrition F1-Score: {f1_score(y_test_at, model_attr.predict(X_test_at)):.4f}")


joblib.dump(attr_preprocessor, os.path.join(MODELS_DIR, "attrition_preprocessor.joblib"))
joblib.dump(model_attr, os.path.join(MODELS_DIR, "model_attrition.joblib"))


print("\n Processing Engine 4 (Salary Growth Evaluation)...")
df_sal = pd.read_csv(SALARY_PATH)


if 'Unnamed: 0' in df_sal.columns:
    df_sal = df_sal.drop(columns=['Unnamed: 0'])


df_sal = df_sal.dropna(subset=['Salary'])

sal_num_features = ['Age', 'Years of Experience']
sal_cat_features = ['Gender', 'Education Level', 'Job Title', 'Country', 'Race']

X_s = df_sal[sal_num_features + sal_cat_features]
y_salary = df_sal['Salary']

sal_preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), sal_num_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), sal_cat_features)
])

X_s_processed = sal_preprocessor.fit_transform(X_s)
X_train_sl, X_test_sl, y_train_sl, y_test_sl = train_test_split(X_s_processed, y_salary, test_size=0.2, random_state=42)


print(" Training Model 4: Gradient Boosting Salary Regressor...")
model_salary = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model_salary.fit(X_train_sl, y_train_sl)
print(f"   -> Salary R² Score: {r2_score(y_test_sl, model_salary.predict(X_test_sl)):.4f}")


joblib.dump(sal_preprocessor, os.path.join(MODELS_DIR, "salary_preprocessor.joblib"))
joblib.dump(model_salary, os.path.join(MODELS_DIR, "model_salary.joblib"))

print("\n" + "="*60 + "\n SUCCESS: All 4 models and preprocessors saved perfectly inside 'models/' directory!")