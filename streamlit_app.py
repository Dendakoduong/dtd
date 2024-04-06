import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load your model here
model = joblib.load('logistic_regression_model.pkl')

# Define numerical_columns
numerical_columns = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age"]

# Load the scaler from the file
scaler = joblib.load('scaler.pkl')

# Define a function to preprocess the input
def preprocess_input(input_data):
    # Convert input_data to DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Apply the same preprocessing steps as you did for your training data
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    return input_df

# Questions
questions = {
    "HighBP": "Have you ever been told by a doctor, nurse or other health professional that you have high blood pressure?",
    "HighChol": "Do you have high cholesterol?",
    "BMI": "BMI",
    "Smoker": "Do you smoke?",#Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]
    "Stroke": "Have you ever diagnosed with a Stroke?",
    "HeartDiseaseorAttack": "Have you ever been reported having coronary heart disease (CHD) or myocardial infarction (MI)?",
    "PhysActivity": "Have you engaged in any physical activity or exercise other than your regular job in the past 30 days?",
    "Fruits": "Do you have the habit of eating fruit one or more times per day? (Do not include juices)",
    "Veggies": "Do you have the habit of eating vegetables one or more times per day?",
    "HvyAlcoholConsump": "Are you a heavy drinker? (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)",
    "GenHlth": "Would you say that in general your health is:",
    "MentHlth": "Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?",
    "PhysHlth": "Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?",
    "DiffWalk": "Do you have serious difficulty walking or climbing stairs?",
    "Sex": "What is your gender?",
    "Age": "Choose your age category",
}

# Define tips
tips = "Here are some tips to reduce the risk of diabetes:\n1. Maintain a healthy diet rich in fruits and vegetables.\n2. Engage in regular physical activity.\n3. Avoid excessive alcohol consumption.\n4. Quit smoking if you smoke.\n5. Monitor your blood pressure and cholesterol levels regularly."

# Define the Streamlit application
def main():
    st.title("Diabetes Risk Prediction")
    st.write("English")
        
    add_selectbox = st.sidebar.selectbox(
        "What do you want to do?",
        ("Predict", "Help", "About")
    )

    if add_selectbox == "Predict":
        features = {
            "HighBP": questions["HighBP"],
            "HighChol": questions["HighChol"],
            "BMI": questions["BMI"],
            "Smoker": questions["Smoker"],
            "Stroke": questions["Stroke"],
            "HeartDiseaseorAttack": questions["HeartDiseaseorAttack"],
            "PhysActivity": questions["PhysActivity"],
            "Fruits": questions["Fruits"],
            "Veggies": questions["Veggies"],
            "HvyAlcoholConsump": questions["HvyAlcoholConsump"],
            "GenHlth": questions["GenHlth"],
            "MentHlth": questions["MentHlth"],
            "PhysHlth": questions["PhysHlth"],
            "DiffWalk": questions["DiffWalk"],
            "Sex": questions["Sex"],
            "Age": questions["Age"],
        }
        
        input_dict = {}
        for feature, description in features.items():
            if feature == "Sex":
                input_dict[feature] = st.radio(questions["Sex"], ("Female", "Male"))
                input_dict[feature] = 1 if input_dict[feature] == "Male" else 0
            elif feature == "Age":
                age_categories = {
                    "18-24": 1,
                    "25-29": 2,
                    "30-34": 3,
                    "35-39": 4,
                    "40-44": 5,
                    "45-49": 6,
                    "50-54": 7,
                    "55-59": 8,
                    "60-64": 9,
                    "65-69": 10,
                    "70-74": 11,
                    "75-79": 12,
                    "80 or older": 13
                }
                selected_age_category = st.selectbox(questions["Age"], list(age_categories.keys()))
                input_dict[feature] = age_categories[selected_age_category]
            elif feature == "BMI":
                height = st.number_input("Enter your height (m)", min_value=1.0, max_value=2.5, step=0.01)
                weight = st.number_input("Enter your weight (kg)", min_value=30.0, max_value=200.0, step=0.1)
                input_dict[feature] = weight / (height ** 2)
            elif feature == "GenHlth":
                gen_hlth_categories = {
                "Poor": 1,
                "Fair": 2,
                "Good": 3,
                "Very good": 4,
                "Excellent": 5
                }
                selected_gen_hlth_category = st.selectbox(questions["GenHlth"], list(gen_hlth_categories.keys()))
                input_dict[feature] = gen_hlth_categories[selected_gen_hlth_category]
            elif feature in ["MentHlth", "PhysHlth"]:
                input_dict[feature] = st.slider(description, 0, 30)
            else:
                input_dict[feature] = st.radio(description, ("No", "Yes"))
                input_dict[feature] = 1 if input_dict[feature] == "Yes" else 0

        # When the user clicks the "Predict" button
        if st.button("Predict"):
            input_df = preprocess_input(input_dict)
            probabilities = model.predict_proba(input_df)[0]
            not_diabetes_prob = probabilities[0] * 100
            diabetes_prob = probabilities[1] * 100

            # Display a message
            st.subheader("Result")
            if diabetes_prob > not_diabetes_prob:
                st.error(f"At risk of diabetes. Probability of having diabetes: {diabetes_prob:.2f}%")
            else:
                #st.success(f"Not at risk of diabetes. Probability of not having diabetes: {not_diabetes_prob:.2f}%")
                diabetes_prob = 100 - not_diabetes_prob
                st.success(f"Not at risk of diabetes. Probability of having diabetes: {diabetes_prob:.2f}%")

            # Create a smaller and transparent pie chart
            plt.figure(figsize=(3, 5), facecolor=(1, 1, 1, 0.5))
            plt.pie([not_diabetes_prob, diabetes_prob], autopct='%1.1f%%')
            plt.title('Diabetes Prediction Probabilities')

            # Add a transparent legend to the right of the pie chart
            plt.legend(['Not Diabetes', 'Diabetes'], bbox_to_anchor=(1, 0.5), loc='center left', framealpha=0.5)

            # Display the pie chart
            st.pyplot(plt.gcf())

            st.subheader("Tips to Reduce Diabetes Risk")
            st.write(tips)
            
    elif add_selectbox == "Help":
        st.info("This app is used to predict the risk of diabetes based on several features. You need to enter the values for these features in the form, and then click 'Predict'.")
    elif add_selectbox == "About":
        st.markdown("These datasets originally came from the <a href='https://www.cdc.gov/brfss/annual_data/annual_data.htm'>CDC's Annual Survey Data</a>. Complete data descriptions can be found in the <a href='https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf'>2015</a> or <a href='https://www.cdc.gov/brfss/annual_data/2021/pdf/codebook21_llcp-v2-508.pdf'>2021</a> codebooks.", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
