import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
import firebase_admin
from firebase_admin import credentials, firestore

#Caching the model and Firebase initialization for faster operations
@st.cache_resource
def load_firebase():
    #Initialize Firebase only if it hasn't been initialized already to avoid duplicate initialization
    if not firebase_admin._apps:
        #Firebase credential from the directory
        cred = credentials.Certificate("disease-overview-firebase-adminsdk-8kos6-fae78f2fdb.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

#Custom CSS for Streamlit app
st.markdown("""
    <style>
        .main-container {
            width: 90%;
            margin: auto;
            padding: 20px;
            background-color: #f0f2f6;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 20px;
        }
        .description, .how-it-works {
            color: #333;
            font-size: 18px;
            text-align: justify;
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .how-it-works-header {
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 20px;
            text-align: justify;
        }
        .content-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            flex-wrap: wrap;
        }
        .column {
            flex: 1;
            margin: 10px;
            max-width: 48%;
        }
        @media screen and (max-width: 1200px) {
            .column {
                max-width: 100%;
                margin-bottom: 20px;
            }
        }
    </style>
""", unsafe_allow_html=True)


#Caching the model for faster predictions
@st.cache_resource
def load_trained_model():
    model = load_model('model_v2.h5')
    return model

#Caching the labels 
def load_class_labels():
    class_labels = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    return class_labels


#Function to fetch disease details from Firestore
def get_disease_details(disease_name, db):
    #Matching disease name with the document name in Firestore
    doc_ref = db.collection('diseases').document(disease_name)  
    doc = doc_ref.get()
    #Condition to implement only if the document exists in Firestore
    if doc.exists:
        disease_data = doc.to_dict()
        name = disease_data.get('name', 'Unknown')
        desc = disease_data.get('desc', 'No description available.')
        symptoms = disease_data.get('symptoms', 'No symptoms available.')
        soln = disease_data.get('soln', 'No control solution available.')
        return name, desc, symptoms, soln
    else:
        return disease_name, "Overview not available.", "Symptoms not available.", "Control solution not available."

#Getting prediction using the trained model
#This function takes 3 inputs: image, model, and class labels
def predict_and_display(img, model, class_labels):
    #Ensuring the image has 3 channels (RGB); if it doesn't, convert the image
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    #Resizing the image to the required size
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    #Prediction with model
    prediction = model.predict(img_array)
    #Prediction index with model
    predicted_class_index = np.argmax(prediction)
    #Prediction class based on index
    predicted_class_label = class_labels[predicted_class_index]
    #Prediction confidence
    confidence_percentage = 100 * np.max(prediction)

    #Returning label and confidence percentage
    return predicted_class_label, confidence_percentage


def main():
    #Streamlit layout with centered header
    st.markdown("<div class='main-container'><div class='header'>FarmAI</div>", unsafe_allow_html=True)

    #How it works section for UI
    st.markdown("<div class='how-it-works-header'><strong>How it Works</strong></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='how-it-works'>
        1. <strong>Upload an Image</strong>: Select an image of a mango leaf showing symptoms of a potential disease.<br>
        2. <strong>Disease Overview</strong>: Once the image is uploaded, the app displays information about the detected disease, including a description and symptoms.<br>
        3. <strong>Compare Symptoms</strong>: Compare the visible symptoms on the leaf with those provided to determine if further action may be needed.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='description'>Upload an image of a mango leaf to view information about the detected disease.</div>", unsafe_allow_html=True)

    #File uploader to upload image with Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    #If an image is uploaded
    if uploaded_file is not None:
        #Initializing Firebase and model
        db = load_firebase()
        model = load_trained_model()
        class_labels = load_class_labels()
        
        image = Image.open(uploaded_file)
        
        st.markdown("<div class='content-container'>", unsafe_allow_html=True)
        
        #Creating two responsive columns
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("<div class='column image-container'>", unsafe_allow_html=True)
            #Displaying the uploaded image and the predicted disease with confidence
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

            
            #Performing disease prediction and display the results
            predicted_class_label, confidence_percentage = predict_and_display(image, model, class_labels)
            
            #Condition where the confidence is low, and can't predict the disease properly for the given image
            if confidence_percentage < 30:
                st.write("Couldn't identify the disease with sufficient confidence from this image. Please upload a clearer image.")
            #When confidence is above threshold
            else:
                st.write(f"Predicted Disease: **{predicted_class_label}**")
                st.write(f"Confidence: **{confidence_percentage:.2f}%**")
            st.markdown("</div>", unsafe_allow_html=True)

        #Column 2 when the confidence is above the set threshold
        if confidence_percentage >= 30:
            with col2:
                st.markdown("<div class='column'>", unsafe_allow_html=True)
                #Fetching disease details from Firestore
                disease_name, overview, symptoms, soln = get_disease_details(predicted_class_label, db)
                
                #Showing the Disease Name on UI
                st.markdown(f"<div class='description'><strong>Disease Name:</strong> {disease_name}</div><br>", unsafe_allow_html=True)
                #Overview
                st.markdown(f"<div class='description'><strong>Overview:</strong> {overview}</div><br>", unsafe_allow_html=True)
                #Symptoms
                st.markdown(f"<div class='description'><strong>Symptoms:</strong> {symptoms}</div><br>", unsafe_allow_html=True)
                #Control
                st.markdown(f"<div class='description'><strong>Control:</strong> {soln}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    #Closing main-container
    st.markdown("</div>", unsafe_allow_html=True)

#Running the main
if __name__ == "__main__":
    main()
