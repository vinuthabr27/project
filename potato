import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Load pre-trained model
MODEL = tf.keras.models.load_model("C:/Users/A/Documents/cv_project/cv_project/training/extra/1.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to convert uploaded file to image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Streamlit app main function
def main():
    st.title("Plant Disease Detection")
    
    st.write("### Upload an image of the plant leaf to predict the disease.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Process the image for prediction
        image_np = read_file_as_image(uploaded_file.read())
        img_batch = np.expand_dims(image_np, 0)  # Add batch dimension
        
        # Make predictions
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        # Display prediction results
        st.write(f"### Predicted Class: **{predicted_class}**")
        st.write(f"### Confidence: **{confidence:.2f}**")

if __name__ == '__main__':
    main()
