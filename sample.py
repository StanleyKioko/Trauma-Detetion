import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Define the class labels
class_labels = ['Healthy', 'Powdery', 'Rust']

# Set page title and favicon
st.set_page_config(page_title='Leaf Disease Detection', page_icon=':leaves:')

# Define custom CSS styles
st.markdown(
    """
    <style>
    .header {
        font-size: 40px;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        color: #228B22;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add title and description
st.markdown('<div class="header">Leaf Disease Detection</div>', unsafe_allow_html=True)
st.write('Upload leaf images to detect their diseases.')

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(['Upload Images', 'Help', 'About'])

with tab1:
    # File uploader for multiple images
    uploaded_files = st.file_uploader('Choose images...', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    
    # Check if files are uploaded
    if uploaded_files:
        # Reverse the order of uploaded files to show the latest image on top
        uploaded_files = list(reversed(uploaded_files))
        
        # Create a container for the results
        results_container = st.container()
        
        # Iterate over each uploaded file
        for uploaded_file in uploaded_files:
            # Perform error handling and validation
            try:
                # Load and preprocess the image
                img = image.load_img(uploaded_file, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = x / 255.0
                
                # Make predictions
                predictions = model.predict(x)
                predicted_class = class_labels[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                # Display the image and prediction in the results container
                with results_container:
                    st.markdown(f'<div class="subheader">{uploaded_file.name}</div>', unsafe_allow_html=True)
                    st.image(img, caption='Uploaded Image', use_column_width=True)
                    st.write(f'Predicted Disease: {predicted_class}')
                    st.write(f'Confidence: {confidence:.2f}%')
                    
                    # Display disease information and recommendations
                    if predicted_class == 'Healthy':
                        st.write('The leaf appears to be healthy.')
                    elif predicted_class == 'Powdery':
                        st.write('The leaf is affected by powdery mildew.')
                        st.write('Symptoms: White powdery spots on leaves and stems.')
                        st.write('Causes: High humidity and poor air circulation.')
                        st.write('Treatment: Remove infected leaves, improve air circulation, and apply fungicides.')
                    elif predicted_class == 'Rust':
                        st.write('The leaf is affected by rust disease.')
                        st.write('Symptoms: Orange or brown pustules on leaves.')
                        st.write('Causes: Fungal infection favored by warm and humid conditions.')
                        st.write('Treatment: Remove infected leaves, improve air circulation, and apply fungicides.')
                    
                    # Visualization of model's confidence
                    fig, ax = plt.subplots()
                    ax.bar(class_labels, predictions[0])
                    ax.set_xlabel('Disease Class')
                    ax.set_ylabel('Probability')
                    ax.set_title('Model Confidence')
                    st.pyplot(fig)
                    
                    # Download button for the result image
                    img_bytes = BytesIO()
                    fig.savefig(img_bytes, format='png')
                    img_bytes.seek(0)
                    st.download_button(
                        label='Download Result',
                        data=img_bytes,
                        file_name=f'{uploaded_file.name}_result.png',
                        mime='image/png'
                    )
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
with tab2:
    st.markdown('<div class="subheader">Help</div>', unsafe_allow_html=True)
    st.write('1. Click on the "Upload Images" tab.')
    st.write('2. Click on the "Choose images..." button and select one or more leaf images.')
    st.write('3. Wait for the app to process the images and display the results.')
    st.write('4. View the predicted disease, confidence score, and additional information for each image.')
    st.write('5. Use the "Download Result" button to download the result image with the visualization.')
    
with tab3:
    st.markdown('<div class="subheader">About</div>', unsafe_allow_html=True)
    st.write('This app is designed to detect diseases in leaf images using a trained machine learning model.')
    st.write('It can identify three types of leaf conditions: Healthy, Powdery, and Rust.')
    st.write('The app provides predictions, confidence scores, and additional information about each detected disease.')
    st.write('It also allows you to download the result image with the visualization of the model\'s confidence.')
    st.write('Please note that the app\'s performance may depend on the quality and clarity of the uploaded images.')