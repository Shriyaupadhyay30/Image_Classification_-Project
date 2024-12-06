import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from lime.lime_image import LimeImageExplainer

#Loading the saved model
model = load_model('model_v2.h5')

#Loading and preprocessing the image for model
img_path = 'D:/Big Data Analytics/Term-2/BDM 3014 - Introduction to Artificial Intelligence 01/Final Project/20211011_133423 (Custom).jpg'
#Resizing image to match input size
img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)
#Adding batch dimension
img_array = np.expand_dims(img_array, axis=0)  

#Preprocessing image as required by EfficientNetB0
img_array_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_array)

#Creating a LIME image explainer
explainer = LimeImageExplainer()

#Generating explanation using LIME
explanation = explainer.explain_instance(
    #Input image
    img_array[0], 
    #Prediction  with model
    model.predict, 
    #Number of top labels to explain (for classification) 
    top_labels=5, 
    #Color to hide (use 0 for default) 
    hide_color=0,
    #Number of random samples to generate for explanation  
    num_samples=1000  
)

#Visualizing the explanation as an image overlay (heatmap)
#Creating two subplots side by side with original image and LIME explainer image
fig, axes = plt.subplots(1, 2, figsize=(15, 10))  

#Original Image
axes[0].imshow(np.array(img)) 
axes[0].set_title('Original Image')
axes[0].axis('off')

#LIME Explanation Heatmap
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], 
    positive_only=True,
    #Number of important features to highlight 
    num_features=10,  
    hide_rest=True
)
#Displaying the original image (for context)
axes[1].imshow(temp)  
#Overlayying the heatmap with plasma colormap
heatmap = axes[1].imshow(mask, cmap='plasma', alpha=0.5)  
axes[1].set_title('LIME Explanation: Heatmap of Important Features')
axes[1].axis('off')

#Adding color bar (legend)
cbar = fig.colorbar(heatmap, ax=axes[1], orientation='vertical')
cbar.set_label('Importance', rotation=270, labelpad=15)

#Printing the LIME explanation details
print("\nLIME Explanation Results:")
print("Top Label for Explanation:", explanation.top_labels[0])
print("Explanation of Important Features:")
for i, (feature, weight) in enumerate(explanation.local_exp[explanation.top_labels[0]]):
    print(f"Feature {i+1}: {feature} with weight: {weight:.4f}")

#Showing the plot
#Adjusting layout to prevent overlay
plt.tight_layout()  
plt.show()