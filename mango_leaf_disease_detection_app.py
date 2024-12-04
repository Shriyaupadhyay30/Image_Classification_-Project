import streamlit as st 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image

# Define the model architecture or load pre-trained model
def define_model():
    # Load the trained model (ensure the correct path to your .h5 file)
    model = load_model('D:/Term 2/Mangoleaf/model_v2.h5')  # Adjust the path as needed
    return model

# Preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize to model's input size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess image for EfficientNet
    return img_array

# Function to predict and display the result
def predict_and_display(image_path, model, class_labels):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    
    # Display the image and prediction
    img = Image.open(image_path)
    st.image(img, caption='Uploaded Leaf Image', use_container_width=True)
    st.markdown(f"### Predicted Disease: **{predicted_class_label}**")
    return predicted_class_label

# Streamlit UI
st.title('Mango Leaf Disease Detection')

# Upload the image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If a file is uploaded, display the prediction
if uploaded_file is not None:
    # Save uploaded file to a temporary location
    image_path = f'./{uploaded_file.name}'
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Define class labels (adjust based on your dataset)
    class_labels = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Die Back", "Gall Midge", 
                    "Healthy", "Powdery Mildew", "Sooty Mould"]

    # Load the trained model
    model = define_model()

    # Predict and display the result
    predicted_disease = predict_and_display(image_path, model, class_labels)

    # Disease overview and symptoms
    disease_info = {
        "Anthracnose": {
            "overview": "A fungal disease caused by Colletotrichum gloeosporioides, leading to dark lesions on leaves, fruits, and stems.",
            "symptoms": [
                "Dark, sunken spots on leaves and fruits",
                "Yellowing and withering of affected areas",
                "Premature fruit drop",
                "Irregularly shaped lesions"
            ]
        },
        "Bacterial Canker": {
            "overview": "Caused by bacterial pathogens, it leads to water-soaked lesions and cracks on the leaf surface.",
            "symptoms": [
                "Water-soaked lesions on leaves",
                "Cracking and splitting of stems",
                "Yellowing and leaf drop",
                "Sticky exudate from affected areas"
            ]
        },
        "Cutting Weevil": {
            "overview": "Damage caused by weevil larvae cutting through leaves and shoots.",
            "symptoms": [
                "Irregular leaf margins",
                "Holes in leaves",
                "Wilting of young shoots",
                "Presence of larvae in damaged areas"
            ]
        },
        "Die Back": {
            "overview": "A fungal disease leading to the drying of twigs and branches from the tip backward.",
            "symptoms": [
                "Drying of twigs and branches",
                "Browning or blackening of affected areas",
                "Reduced flowering and fruiting",
                "Premature leaf fall"
            ]
        },
        "Gall Midge": {
            "overview": "Damage caused by insect larvae forming galls on leaves.",
            "symptoms": [
                "Swollen, abnormal growths on leaves (galls)",
                "Yellowing and wilting of affected leaves",
                "Deformation of young leaves",
                "Stunted growth"
            ]
        },
        "Healthy": {
            "overview": "The leaf appears to be healthy with no visible signs of disease.",
            "symptoms": ["No symptoms detected."]
        },
        "Powdery Mildew": {
            "overview": "A fungal disease characterized by white, powdery patches on leaves and stems.",
            "symptoms": [
                "White, powdery spots on leaves and stems",
                "Distorted or stunted growth",
                "Yellowing of leaves",
                "Leaf drop in severe cases"
            ]
        },
        "Sooty Mould": {
            "overview": "A fungal growth on leaf surfaces caused by honeydew excreted by insects.",
            "symptoms": [
                "Black, soot-like coating on leaves and stems",
                "Reduced photosynthesis due to coating",
                "Yellowing of leaves",
                "Leaf drop in severe infestations"
            ]
        }
    }

    # Show disease details
    if predicted_disease in disease_info:
        st.markdown(f"**Overview:** {disease_info[predicted_disease]['overview']}")
        st.markdown("**Symptoms:**")
        for symptom in disease_info[predicted_disease]["symptoms"]:
            st.markdown(f"- {symptom}")
    else:
        st.markdown("No information available for this disease.")
