import os
import pandas as pd

DATASET_BASE = "datasets"
PATHS = {
    "Attrition Engine": os.path.join(DATASET_BASE, "dataset1_attrition", "ibm_attrition.csv"),
    "Promotion/Perf Engine": os.path.join(DATASET_BASE, "dataset2_promotion", "employee_promotion.csv"),
    "Salary Engine": os.path.join(DATASET_BASE, "dataset3_salary", "employee_salary.csv")
}

def check_and_summarize_data():
    print(" Initializing KARMA-AI Data Verification Pipeline...\n" + "="*50)
    
    for engine_name, path in PATHS.items():
        if not os.path.exists(path):
            print(f" CRITICAL ERROR: Could not find file for [{engine_name}] at path: {path}")
            print(" Please ensure folders are named exactly and files match the naming spec.\n")
            continue
            
        print(f" Found data asset for [{engine_name}]")
       
        df = pd.read_csv(path)
        print(f"   -> Records (Rows): {df.shape[0]} | Attributes (Columns): {df.shape[1]}")
        
       
        print(f"   -> Sample Columns: {list(df.columns[:5])}...")
        
       
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            print(f"   ->  Contains {null_counts} total missing data points (will be handled by preprocessing pipeline).")
        else:
            print("   -> Clean line state: No missing data detected.")
        print("-" * 50)

if __name__ == "__main__":
    check_and_summarize_data()