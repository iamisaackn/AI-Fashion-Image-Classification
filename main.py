# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from rembg import remove 

# Set the working directory and model path
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/model/trained_fashion_mnist_model.keras"

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Remove background
    image_no_bg = remove(image)
    # Convert to grayscale
    img_gray = image_no_bg.convert('L')
    # Resize the image to 28x28 pixels
    img_resize = img_gray.resize((28, 28))
    # Convert the image to a numpy array and normalize pixel values
    img_array = np.array(img_resize) / 255.0
    # Expand dimensions to match the input shape expected by the model
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded file's bytes
    image = Image.open(uploaded_image)
    
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100, 100))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image
            img_array = preprocess_image(image)

            # Make a prediction using the pre-trained model
            result = model.predict(img_array)
            predicted_class = np.argmax(result)
            prediction = class_names[predicted_class]

            st.success(f'Prediction: {prediction}')
