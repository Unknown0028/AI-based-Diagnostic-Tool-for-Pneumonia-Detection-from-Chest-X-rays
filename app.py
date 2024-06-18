import streamlit as st
import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np
from torchvision import transforms

# Load the pre-trained CheXNet model
model = xrv.models.DenseNet(weights="all")

# Function to preprocess the image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to predict pneumonia
def model_predict(img, model):
    model.eval()
    with torch.no_grad():
        output = model(img)
    output = torch.sigmoid(output).numpy()
    return output

# Streamlit UI
st.title("AI-based Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image and the model will predict if it shows signs of pneumonia.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png", "dcm"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Chest X-ray.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_tensor = preprocess_image(image)

    # Predict the image
    preds = model_predict(img_tensor, model)
    pneumonia_prob = preds[0][xrv.datasets.default_pathologies.index('Pneumonia')]
    st.write(f"Prediction: Pneumonia Probability: {pneumonia_prob:.4f}")

# To run the Streamlit app, use the command below:
# streamlit run app.py
