import pandas as pd
import streamlit as st
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Load the classification model
model_classification = joblib.load('model_and_preprocessor.pkl')

# Create Sidebar to navigate between EDA and Classification
sidebar = st.sidebar
mode = sidebar.radio('Mode', ['EDA', 'Classification'])

if mode == "EDA":

    def main():

        # Header of Customer Satisfaction Prediction
        html_temp = """
                    <div style="background-color:#F5F5F5">
                    <h1 style="color:#31333F;text-align:center;">Customer Satisfaction Prediction</h1>
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)
        
        # Create sidebar to upload CSV files
        with st.sidebar.header('Upload your CSV data'):
            uploaded_file = st.sidebar.file_uploader('Upload your input csv file')

        if uploaded_file is not None:
            # Read file
            EDA_sample = pd.read_csv(uploaded_file)
            st.header('**Input DataFrame**')
            st.write(EDA_sample)
            st.write('---')

            # Generate the Pandas Profiling Report
            pr = ProfileReport(EDA_sample, explorative=True)
            st.header('**Pandas Profiling Report**')
            st_profile_report(pr)
        
        else:
            st.info('Awaiting for CSV file to be uploaded.')

    if __name__ == '__main__':
        main()

if mode == "Classification":

    # Define function to predict classification based on assigned features
    def predict_satisfaction(price, estimated_vs_actual_shipping, order_value, order_month, order_hour, payment_sequential):

        prediction_classification = model_classification.predict(pd.DataFrame({'price': [price], 
                                                                              'estimated_vs_actual_shipping': [estimated_vs_actual_shipping], 
                                                                              'order_value': [order_value], 
                                                                              'order_month': [order_month], 
                                                                              'order_hour': [order_hour], 
                                                                              'payment_sequential': [payment_sequential]}))
        return prediction_classification

    def main():

        # Header of Customer Satisfaction Prediction
        html_temp = """
                    <div style="background-color:#F5F5F5">
                    <h1 style="color:#31333F;text-align:center;">Customer Satisfaction Prediction</h1>
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html=True)
        
        # Assign all features with desired data input method
        price = st.number_input('Price', value=0.0)
        estimated_vs_actual_shipping = st.number_input('Estimated vs Actual Shipping Days', value=0)
        order_value = st.number_input('Order Value', value=0.0)
        order_month = st.number_input('Order Month', value=1, min_value=1, max_value=12)
        order_hour = st.number_input('Order Hour', value=0, min_value=0, max_value=23)
        payment_sequential = st.number_input('Payment Sequential', value=1, min_value=1)
        
        result = ''

        # Predict Customer Satisfaction
        if st.button('Predict Satisfaction'):
            result = predict_satisfaction(price, estimated_vs_actual_shipping, order_value, order_month, order_hour, payment_sequential)
            if result == 0:
                result = 'Not Satisfied'
                st.success(f'The Customer is {result}')
            else:
                result = 'Satisfied'
                st.success(f'The Customer is {result}')

    if __name__ == '__main__':
        main()
