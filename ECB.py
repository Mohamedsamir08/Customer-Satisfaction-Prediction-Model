import streamlit as st
import joblib
import numpy as np

# Load your trained pipeline
model = joblib.load('final_ECommerce_model.pkl')

# Define the structure of your app
def main():
    st.title('Customer Satisfaction Prediction App')

   # Define inputs with appropriate ranges and default values based on your data
    estimated_vs_actual_shipping = st.number_input('Estimated vs Actual Shipping Days', min_value=-189, max_value=146, value=11)
    time_to_delivery = st.number_input('Time to Delivery', min_value=-7, max_value=208, value=9)
    payment_value = st.number_input('Payment Value', min_value=0.0, max_value=13664.08, value=107.78)
    late_delivery = st.number_input('Late Delivery', min_value=0, max_value=1, value=0) 

# Prediction button
    if st.button('Predict Satisfaction'):
        # Create an array with the input data
        # Make sure all inputs are included in the array in the correct order
        input_data = np.array([[estimated_vs_actual_shipping, time_to_delivery, payment_value, late_delivery]])

        # Get the prediction
        prediction = model.predict(input_data)

        # Output the prediction
        if prediction[0] == 1:
            st.success('The customer is satisfied.')
        else:
            st.error('The customer is not satisfied')

if __name__ == '__main__':
    main()