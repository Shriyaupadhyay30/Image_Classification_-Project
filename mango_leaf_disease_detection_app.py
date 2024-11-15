import streamlit as st
from PIL import Image

# Custom CSS for styling
st.markdown("""
    <style>
        .main-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .header {
            font-size: 28px;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 20px 0;
        }
        .description, .how-it-works {
            color: #333;
            font-size: 18px;
        }
        .symptoms-list {
            padding-left: 20px;
            font-size: 16px;
        }
        .image-container {
            text-align: center;
        }
        .how-it-works-header {
            font-size: 22px;
            font-weight: bold;
            color: #333;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to provide a general disease overview
def disease_overview():
    disease_name = "Anthracnose"
    overview = (
        "Anthracnose is a fungal disease commonly affecting mango leaves. "
        "It is caused by the fungus Colletotrichum gloeosporioides. The disease "
        "leads to dark lesions on leaves, fruits, and stems. Symptoms include "
        "black spots, yellowing, and premature fruit drop, which can severely "
        "impact yield and quality."
    )
    symptoms = [
        "Dark, sunken spots on leaves and fruits",
        "Yellowing and withering of affected areas",
        "Premature fruit drop",
        "Irregularly shaped lesions"
    ]
    return disease_name, overview, symptoms

# Streamlit UI
st.markdown("<div class='header'>FarmAI</div>", unsafe_allow_html=True)

# "How it Works" section
st.markdown("<div class='how-it-works-header'><strong>How it Works</strong></div>", unsafe_allow_html=True)
st.markdown("""
<div class='how-it-works'>
    1. <strong>Upload an Image</strong>: Select an image of a mango leaf showing symptoms of a potential disease.<br>
    2. <strong>Disease Overview</strong>: Once the image is uploaded, the app displays information about the detected disease, including a description and symptoms.<br>
    3.<strong>Compare Symptoms</strong>: Compare the visible symptoms on the leaf with those provided to determine if further action may be needed.
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='description'>Upload an image of a mango leaf to view information about the detected disease.</div>", unsafe_allow_html=True)

# File uploader to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Show image and display disease overview side-by-side
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the uploaded image on the left with centered alignment
        st.markdown("<div class='image-container'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Display the disease overview on the right with enhanced styling
        disease_name, overview, symptoms = disease_overview()
        
        st.markdown(f"<div class='description'><strong>Disease Name:</strong> {disease_name}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='description'><strong>Overview:</strong> {overview}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='description'><strong>Symptoms:</strong></div>", unsafe_allow_html=True)
        st.markdown("<ul class='symptoms-list'>", unsafe_allow_html=True)
        for symptom in symptoms:
            st.markdown(f"<li>{symptom}</li>", unsafe_allow_html=True)
        st.markdown("</ul>", unsafe_allow_html=True)
