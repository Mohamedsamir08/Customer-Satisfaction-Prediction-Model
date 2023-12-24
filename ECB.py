
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
    order_month = st.number_input('Order Month', min_value=1, max_value=12, value=6)
    order_hour = st.number_input('Order Hour', min_value=0, max_value=23, value=15)
    price = st.number_input('Price', min_value=0.85, max_value=6735.0, value=120.0)
    payment_sequential = st.number_input('Payment Sequential', min_value=1, max_value=26, value=1)
    order_value = st.number_input('Order Value', min_value=6.08, max_value=6929.31, value=140.0)

    # Prediction button
    if st.button('Predict Satisfaction'):
        # Create an array with the input data
        input_data = np.array([[estimated_vs_actual_shipping, order_month, order_hour, price, payment_sequential, order_value]])

        # Get the prediction
        prediction = model.predict(input_data)

        # Output the prediction
        if prediction[0] == 1:
            st.success('The customer is likely satisfied.')
        else:
            st.error('The customer is likely not satisfied.')

if __name__ == '__main__':
    main()
