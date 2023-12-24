import streamlit as st
import joblib
import pandas as pd

# Load the classification model
model_classification = joblib.load('model_and_preprocessor.pkl')

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
        # Define function to predict classification based on assigned features
        def predict_satisfaction(price, estimated_vs_actual_shipping, order_value, order_month, order_hour, payment_sequential):

            prediction_classification = model_classification.predict(pd.DataFrame({'price': [price], 
                                                                                  'estimated_vs_actual_shipping': [estimated_vs_actual_shipping], 
                                                                                  'order_value': [order_value], 
                                                                                  'order_month': [order_month], 
                                                                                  'order_hour': [order_hour], 
                                                                                  'payment_sequential': [payment_sequential]}))
            return prediction_classification

        result = predict_satisfaction(price, estimated_vs_actual_shipping, order_value, order_month, order_hour, payment_sequential)
        if result == 0:
            result = 'Not Satisfied'
            st.success(f'The Customer is {result}')
        else:
            result = 'Satisfied'
            st.success(f'The Customer is {result}')

if __name__ == '__main__':
    main()
