# project_monitoring.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------------------
# Section 1: Data Preparation
# --------------------------
# Dummy dataset for risk prediction:
# Features: supply_delay, workforce_shortage, weather_disruption, material_cost_fluct
# Label: risk (0 = Low Risk, 1 = High Risk)

# Path to your dataset file
dataset_path = r"C:\Users\sinha\OneDrive\Desktop\createch\final\dataset.csv"  # Use 'r' to handle Windows path properly

# Load dataset from CSV
try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")


# --------------------------
# Section 2: Model Training for Risk Prediction
# --------------------------
X = df[['supply_delay', 'workforce_shortage', 'weather_disruption', 'material_cost_fluct']]
y = df['risk']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Optional model evaluation
y_pred = model.predict(X_test)
print("Model Evaluation on Test Data:")
print(classification_report(y_test, y_pred))

# --------------------------
# Section 3: Supplier Risk Assessment Simulation
# --------------------------
# Simulated supplier performance metrics
# Here we assume each supplier has three metrics:
#   - on_time_delivery (percentage)
#   - avg_delay (in days)
#   - supplier_rating (scale 1-10)
supplier_data = {
    'supplier': ['Supplier A', 'Supplier B', 'Supplier C'],
    'on_time_delivery': [95, 80, 70],
    'avg_delay': [1, 3, 5],
    'supplier_rating': [9, 7, 6]
}
supplier_df = pd.DataFrame(supplier_data)
# Calculate a simple risk score: lower on-time delivery, higher avg_delay, lower rating increases risk
supplier_df['risk_score'] = (100 - supplier_df['on_time_delivery']) + supplier_df['avg_delay']*2 + (10 - supplier_df['supplier_rating'])*3

# Determine supplier risk status
supplier_df['risk_status'] = supplier_df['risk_score'].apply(lambda x: 'High Risk' if x > 20 else 'Low Risk')

# --------------------------
# Section 4: Streamlit Dashboard
# --------------------------
st.title("Enhanced Project Execution Risk Monitoring Dashboard")

st.markdown("""
This dashboard predicts project execution risk and provides automated mitigation suggestions. 
It also simulates supplier risk assessment, real-time workforce/equipment tracking, and cost/resource efficiency insights.
""")

# Input widgets for primary project parameters
supply_delay = st.number_input("Supply Chain Delay (in days)", min_value=0, max_value=30, value=3)
workforce_shortage = st.slider("Workforce Shortage (0-10 scale)", 0, 10, 3)
weather_disruption = st.slider("Weather Disruption (0-10 scale)", 0, 10, 2)
material_cost_fluct = st.number_input("Material Cost Fluctuation (in %)", min_value=0, max_value=100, value=5)

# Prepare input data for risk prediction
input_data = pd.DataFrame({
    'supply_delay': [supply_delay],
    'workforce_shortage': [workforce_shortage],
    'weather_disruption': [weather_disruption],
    'material_cost_fluct': [material_cost_fluct]
})

if st.button("Predict Risk"):
    # Predict risk using the trained model
    prediction = model.predict(input_data)[0]
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    
    st.subheader(f"Predicted Project Risk: **{risk_label}**")
    
    if prediction == 1:
        st.error("ALERT: High risk detected! Please review project plans and mitigation strategies.")
        st.markdown("### Recommended Mitigation Actions:")
        st.markdown("- **Increase workforce by 10%** to address labor shortages.")
        st.markdown("- **Secure additional raw materials** to counter supply delays.")
        st.markdown("- **Evaluate supplier performance:** Check supplier risk scores and consider alternatives if needed.")
        st.markdown("- **Automated task reallocation:** Use AI-driven scheduling to optimize workforce and equipment usage.")
        st.markdown("- **Initiate cost-impact analysis:** Estimate savings from mitigating these risks.")
    else:
        st.success("Project risk is low. Continue monitoring as planned.")

# --------------------------
# Section 5: Simulated Real-Time Data & Automation Enhancements
# --------------------------
st.markdown("---")
st.markdown("### Simulated Real-Time Data & Automation Features")

# Real-Time Workforce & Equipment Tracking Simulation
if st.checkbox("Enable Real-Time Workforce & Equipment Data Simulation"):
    real_time_workforce = np.random.randint(80, 100)  # Simulated workforce attendance percentage
    equipment_utilization = np.random.randint(60, 95)  # Simulated equipment utilization percentage
    st.info(f"Real-Time Workforce Attendance: **{real_time_workforce}%**")
    st.info(f"Equipment Utilization: **{equipment_utilization}%**")
    
    # Automated Workforce Scheduling & Deployment Suggestion
    if workforce_shortage > 7 or real_time_workforce < 85:
        st.warning("**Automated Workforce Scheduling Suggestion:** Consider reassigning shifts or hiring temporary workers.")
    # Equipment Downtime Prediction & Utilization Optimization Suggestion
    if equipment_utilization < 70:
        st.warning("**Equipment Utilization Alert:** Schedule maintenance or reassign equipment to optimize usage.")

# Supplier Risk Assessment Display
st.markdown("---")
st.markdown("### Supplier Risk Assessment")
st.dataframe(supplier_df[['supplier', 'on_time_delivery', 'avg_delay', 'supplier_rating', 'risk_score', 'risk_status']])
st.markdown("**Recommendation:** For suppliers marked as **High Risk**, evaluate alternative sources or negotiate improved delivery terms.")

# Simulated Task Scheduling & Automation Integration
st.markdown("---")
st.markdown("### Task Scheduling & Cost Efficiency Recommendations")
if st.button("Show Automated Task Scheduling Recommendations"):
    st.markdown("- **Task Scheduling Automation:** Integrate with tools like Microsoft Project or Asana to auto-adjust schedules based on risk.")
    st.markdown("- **AI-Driven Work Reallocation:** When risk is high, consider reassigning tasks to maximize productivity.")
    st.markdown("- **Cost & Resource Efficiency:** Integrate with financial software to track real-time costs and optimize resource allocation.")

st.markdown("---")
st.markdown("**Note:** This prototype uses simulated data and recommendations. In a production system, replace dummy data with real-time inputs via APIs and integrate with IoT and ERP systems for full automation and decision support.")
