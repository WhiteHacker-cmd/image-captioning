import streamlit as st
from PIL import Image
import io
import requests

from predict import ImgCModel

# Import your image captioning model and other necessary libraries here
# Replace 'your_captioning_model' with your actual model loading code

# Define the Streamlit app

IM = ImgCModel()

def main():
    st.title("Image Captioning Web App")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        new_image = image.resize((600, 400))
        if st.button('Generate Caption'):
            with st.spinner("Generating caption..."):
                # Preprocess the image and generate a caption using your model
                caption = IM.predict(uploaded_image)
        


                # Display the generated caption
                st.success("Caption: " + caption)
        st.image(new_image, caption="Uploaded Image", use_column_width=True)

        # Provide a loading animation
        

# Function to generate caption using your model
def generate_caption(image):
    # Replace this with your model code to generate a caption
    # You'll need to preprocess the image and use your model to get the caption
    # Example: caption = your_captioning_model.generate_caption(image)
    caption = "A sample caption for the image."
    return caption

if __name__ == "__main__":
    main()
