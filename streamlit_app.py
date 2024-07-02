import numpy as np
import pandas as pd

import streamlit as st

from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment

from zenml.services.service_status import ServiceState


def main():
    st.title("Customer Satisfaction Pipeline")
    st.markdown(
        """#### Problem Statement 
The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc.

#### Description of Features 
This app is designed to predict the customer satisfaction score for a given customer. You can input the features of the product listed below and get the customer satisfaction score. 
| Models        | Description   | 
| ------------- | -     | 
| Payment Sequential | Customer may pay an order with more than one payment method. If he does so, a sequence will be created to accommodate all payments. | 
| Payment Installments   | Number of installments chosen by the customer. |  
| Payment Value |       Total amount paid by the customer. | 
| Price |       Price of the product. |
| Freight Value |    Freight value of the product.  | 
| Product Name length |    Length of the product name. |
| Product Description length |    Length of the product description. |
| Product photos Quantity |    Number of product published photos |
| Product weight measured in grams |    Weight of the product measured in grams. | 
| Product length (CMs) |    Length of the product measured in centimeters. |
| Product height (CMs) |    Height of the product measured in centimeters. |
| Product width (CMs) |    Width of the product measured in centimeters. |
"""
    )

    payment_sequential = st.sidebar.slider("Payment Sequential")
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.number_input("Payment Value")
    price = st.number_input("Price")
    freight_value = st.number_input("freight_value")
    product_name_length = st.number_input("Product name length")
    product_description_length = st.number_input("Product Description length")
    product_photos_qty = st.number_input("Product photos Quantity ")
    product_weight_g = st.number_input("Product weight measured in grams")
    product_length_cm = st.number_input("Product length (CMs)")
    product_height_cm = st.number_input("Product height (CMs)")
    product_width_cm = st.number_input("Product width (CMs)")

    predict_btn = st.button("Predict")
    
    
    if predict_btn:
        try:
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                running=False
            )
        except Exception as e:
            print(f"Error located prediction service: {e}")
            service = None
        
        if service is None or service.status.state == ServiceState.INACTIVE:
            st.write("No prediction service available. Running deployment pipeline...")
            run_deployment(["--config", "deploy_and_predict", "--data-path", "./data/olist_merged.csv"], standalone_mode=False)
            service = prediction_service_loader(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step",
                running=False
            )

        df = pd.DataFrame(
            {
                
                "price": [price],
                "freight_value": [freight_value],
                "payment_sequential": [payment_sequential],
                "payment_installments": [payment_installments],
                "payment_value": [payment_value],
                "product_name_lenght": [product_name_length],
                "product_description_lenght": [product_description_length],
                "product_photos_qty": [product_photos_qty],
                "product_weight_g": [product_weight_g],
                "product_length_cm": [product_length_cm],
                "product_height_cm": [product_height_cm],
                "product_width_cm": [product_width_cm],
            }
        )

        prediction = service.predict(df)

        st.success(f"Predicted Customer Satisfactory rate (0-5): {prediction}")
        

if __name__ == "__main__":
    main()

