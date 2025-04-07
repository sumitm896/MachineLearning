import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load trained Polynomial Regression model
model = joblib.load("polynomial_ridge_model.pkl")

# Streamlit UI: Title and Introduction
st.title("🌊 Flood Prediction Dashboard")
st.write("Enter the risk factors below to predict flood probability.")

# Sidebar Inputs: User-Friendly Number Inputs with Tooltips
st.sidebar.header("Enter Risk Factors")
feature_info = {
    "DeterioratingInfrastructure": "Quality of infrastructure over time.",
    "MonsoonIntensity": "Strength of the monsoon season.",
    "DamsQuality": "Maintenance and condition of dams.",
    "TopographyDrainage": "Efficiency of natural drainage systems.",
    "Siltation": "Amount of silt accumulation in water bodies.",
    "RiverManagement": "Management practices of river resources.",
    "PopulationScore": "Impact of population density on flood risk.",
    "Landslides": "Frequency and severity of landslides.",
    "ClimateChange": "Effects of changing climate patterns.",
    "Deforestation": "Rate of tree cover loss.",
    "AgriculturalPractices": "Influence of farming methods on land.",
    "WetlandLoss": "Reduction in natural wetland areas.",
    "IneffectiveDisasterPreparedness": "Readiness for flood disasters.",
    "PoliticalFactors": "Political decisions affecting flood management.",
    "Watersheds": "Health and status of watershed areas.",
    "InadequatePlanning": "Effectiveness of urban planning.",
    "Urbanization": "Impact of urban sprawl.",
    "DrainageSystems": "Quality of man-made drainage systems.",
    "Encroachments": "Extent of unauthorized land use.",
    "CoastalVulnerability": "Risk to coastal areas."
}

user_inputs = {}
# Create two columns in the sidebar for a cleaner layout
cols = st.sidebar.columns(2)
i = 0
for feature, tooltip in feature_info.items():
    col = cols[i % 2]
    user_inputs[feature] = col.number_input(
        label=feature,
        min_value=0,
        max_value=10,
        value=5,
        step=1,
        help=tooltip
    )
    i += 1

# Prediction Section: When user clicks the button, make a prediction
if st.button("Predict Flood Probability"):
    input_values = np.array(list(user_inputs.values())).reshape(1, -1)
    prediction = model.predict(input_values)
    st.write(f"🌊 **Predicted Flood Probability:** `{prediction[0]:.2f}`")

# Explanation Section
st.write("""
---
**How This Works:**  
✅ The model analyzes **20 risk factors** and predicts the likelihood of a flood.  
✅ Adjust the values in the **sidebar** and click **Predict**.  
✅ The **higher the probability**, the greater the risk of flooding.
""")

#############################################
# Data Analysis Section: Correlation & Boxplot
#############################################
st.write("---")
st.header("Data Analysis")

@st.cache(allow_output_mutation=True)
def load_data():
    # Load a sample of the dataset for analysis (adjust nrows for performance)
    df = pd.read_csv("train.csv", nrows=30000)
    return df

df_sample = load_data()

# Correlation Matrix
st.subheader("Correlation Heatmap")
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
corr_matrix = df_sample.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# Boxplot for Flood Probability
st.subheader("Boxplot of Flood Probability")
fig_box, ax_box = plt.subplots(figsize=(6, 4))
sns.boxplot(y=df_sample["FloodProbability"], ax=ax_box)
ax_box.set_title("Flood Probability Distribution")
st.pyplot(fig_box)
