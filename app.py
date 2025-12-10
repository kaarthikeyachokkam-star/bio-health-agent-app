import streamlit as st
import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))

def heart_disease_prediction(input_data):
    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
      return 'The Person does not have a Heart Disease'
    else:
      return 'The Person has Heart Disease'


def main():
    st.title('Heart Disease Prediction Web App')

    # Getting the input data from the user
    age = st.text_input('Age')
    sex = st.text_input('Sex (0 = female, 1 = male)')
    cp = st.text_input('Chest Pain Type (0-3)')
    trestbps = st.text_input('Resting Blood Pressure')
    chol = st.text_input('Serum Cholestoral in mg/dl')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    thalach = st.text_input('Maximum Heart Rate Achieved')
    exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.text_input('ST depression induced by exercise relative to rest')
    slope = st.text_input('The slope of the peak exercise ST segment (0-2)')
    ca = st.text_input('Number of major vessels (0-4) colored by flourosopy')
    thal = st.text_input('Thal (0-3)')

    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            input_list = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
            diagnosis = heart_disease_prediction(input_list)
        except ValueError:
            diagnosis = "Please enter valid numeric inputs for all fields."
        
    st.success(diagnosis)


if __name__ == '__main__':
    main()