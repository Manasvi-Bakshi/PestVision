import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

# Define the CNN model (PestCNN)
class PestCNN(nn.Module):
    def __init__(self, num_classes):
        super(PestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = PestCNN(num_classes=17)
model.load_state_dict(torch.load("pest_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Pest descriptions dictionary
pest_descriptions = {
    'Spodoptera_litura': "Spodoptera litura, also known as the cotton leafworm, is a destructive pest commonly found in tropical regions. It affects a wide range of crops, causing damage by feeding on leaves and reducing plant productivity.",
    'aphid': "Aphids are small sap-sucking insects that attack various plants, causing stunted growth and transmitting plant viruses. They reproduce rapidly, leading to infestations if not controlled in time.",
    'beet_armyworm': "The beet armyworm is a caterpillar pest that primarily targets crops like beet, spinach, and cotton. It feeds on the leaves, leaving them ragged, and may also attack the fruits or flowers.",
    'borer': "Borers are insects whose larvae tunnel into the stems, trunks, or fruits of plants, causing significant damage. They are a serious pest for crops like sugarcane, maize, and fruit trees.",
    'chemical_fertilizer': "Chemical fertilizers provide essential nutrients to crops but may also have unintended effects on pest populations. Managing fertilizer use is crucial to avoid promoting pest outbreaks.",
    'cnidocampa_flavescens': "Cnidocampa flavescens is a species of slug moth whose larvae can cause defoliation in trees. The caterpillars possess irritating spines that can harm humans and animals.",
    'corn_borer': "Corn borers are pests that attack maize and other crops, boring into the stems and ears. They reduce crop yield and quality by weakening the plant structure.",
    'cotton_bollworm': "The cotton bollworm is a highly destructive pest for cotton and other crops. Its larvae feed on the leaves, flowers, and fruits, causing significant economic losses.",
    'fhb': "Fusarium head blight (FHB) is a fungal disease affecting cereal crops, especially wheat. It produces toxins that make the grain unsafe for consumption and leads to reduced yield.",
    'grasshopper': "Grasshoppers are voracious feeders that damage crops by chewing on leaves, stems, and even fruits. Large swarms can devastate entire fields in a short time.",
    'longhorn_beetle': "Longhorn beetles are wood-boring insects that can damage trees and wooden structures. Their larvae bore into the wood, compromising the structural integrity of the host plant.",
    'oriental_fruit_fly': "The oriental fruit fly is a major pest for fruit crops, laying eggs inside the fruit. The larvae feed on the fruit pulp, causing rot and making the fruit unmarketable.",
    'pesticides': "Pesticides are chemicals used to control pests. While effective, they must be used carefully to avoid harming beneficial insects, humans, and the environment.",
    'plutella_xylostella': "The diamondback moth, or Plutella xylostella, is a pest of cruciferous crops like cabbage. Its larvae cause damage by feeding on the leaves, often making them skeletonized.",
    'rice_planthopper': "Rice planthoppers are sap-feeding insects that damage rice plants by sucking out nutrients. They also transmit viral diseases that can severely reduce rice yields.",
    'rice_stem_borer': "Rice stem borers are larvae that bore into the rice stems, causing 'dead heart' or 'white head' symptoms. They are one of the most damaging pests for rice cultivation.",
    'rolled_leaf_borer': "Rolled leaf borers cause damage to crops like rice by rolling and feeding on the leaves. This pest affects photosynthesis and reduces crop growth and yield."
}

# Streamlit app
st.set_page_config(page_title="PestVision", page_icon="🌾")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Pest Recognition"])

# Home Page
if app_mode == "Home":
    st.title("🌾 PestVision")
    st.image("home.jpeg", use_column_width=True, caption="Protect Your Crops")
    st.markdown("""

    Welcome to **PestVision**! 🌱🐛🔍

    Our goal is to help identify and manage agricultural pests effectively. Upload an image of a pest, and our system will analyze it to recognize the pest type. Protect your crops and ensure a healthier harvest.

    ### How It Works
    1. **Upload Image:** Go to the **Pest Recognition** page and upload an image of a pest.
    2. **Analysis:** Our system will process the image using advanced algorithms to detect potential pests.
    3. **Results:** View the results, including pest identification and further resources.

    ### Project Code
    Check out the [GitHub repository](https://github.com/Manasvi-Bakshi/PestVision) for more details.
    """)

# About Page
elif app_mode == "About":
    st.title("About PestVision")
    st.write("""
    This project utilizes a convolutional neural network (CNN) for agricultural pest classification. It helps farmers and agricultural experts quickly identify pests. We created this project in our Software Project Management Lab.
    """)
    st.header("Research Paper")
    st.header("PEST DETECTION AND CLASSIFICATION IN FRUITS AND VEGETABLE")
    st.write("Our research paper can be viewed below:")
    st.write(""" **Abstract**:  
              Crop losses due to pests can devastate agriculture, with up to 30 percent of yields lost yearly. This paper presents a comprehensive review of existing techniques for pest detection in fruits and vegetables using computer vision and deep learning models. This study aims
        to enhance pest management and boost crop protection by improving accuracy and reducing reliance on manual methods. Our review highlights innovative approaches in 
        AI-driven pest identification, setting the stage for more creative, more efficient agricultural practices.""")
    st.write("**Keywords**:  *Pest Detection and Classification, YOLO Architecture, Deep Learning in Agriculture, Convolutional Neural Networks (CNNs)*")
    st.write(""" **Introduction**:    
            Agricultural production faces a substantial challenge from pests, which inflict considerable harm to fruits and vegetables, impacting both crop quality and yield. To
            safeguard financial interests and ensure food availability, efficient pest management is crucial. In recent times, deep learning, particularly in the realm of computer vision, has
            shown remarkable promise in tackling these issues by enabling precise pest identification and categorization, even with limited datasets.  
            This study offers a thorough examination of the latest innovations in deep learning methodologies for identifying and categorizing pests in fruits and vegetables. The
            research explores various models, such as convolutional neural networks (CNNs), vision transformers, and few-shot learning approaches, which have proven effective in
            creating accurate pest recognition systems. Techniques like MobileNet CNN with autoencoder, ResNet-50, DM-YOLOv8, and enhanced versions of YOLOv7 and
            Cascade R-CNN have boosted detection capabilities. Additionally, methods such as PotatoPestNet, EfficientNet, ConvNeXt, and LAD-Net, along with optimized pretrained CNN models, further enhance accuracy.  
            The study also delves into innovative strategies, including the use of YOLOv5 for UAV imagery, U-Net MobileNet combined with YOLOv5, and hybrid models that merge
            CNN architectures with majority voting. Other noteworthy techniques, such as IResNet with logistic regression, tiny YOLOv3 for drone-based images, VGG16, InceptionV3,
            and CNNs paired with SLIC SuperPixel segmentation, showcase the broad spectrum of deep learning applications in this field. These advancements contribute to the creation
            of more efficient, scalable, and accurate pest detection systems, which are vital for sustainable farming practices""")
    
    st.write("If you wish to read the entire paper you can download it from below:")
    with open("Pest Fruits & Vegetable.pdf", "rb") as pdf_file:
        st.download_button("Download PDF", pdf_file, "Pest Fruits & Vegetable.pdf")
    st.header("Group Members")
    st.markdown("- Manasvi Bakshi")
    st.markdown("- Prabhijit Kaur")
    st.markdown("- Om Tiwari")
    st.markdown("- Kanv Chugh")
    

# Pest Recognition Page
elif app_mode == "Pest Recognition":
    st.title("Pest Recognition and Classification")
    uploaded_file = st.file_uploader("Choose an image to upload:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

        input_image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_image)
            predicted_probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(predicted_probabilities.data, 1)
            predicted_label = predicted.item()

        label_names = [
            'Spodoptera_litura', 'aphid', 'beet_armyworm', 'borer', 
            'chemical_fertilizer', 'cnidocampa_flavescens', 'corn_borer', 
            'cotton_bollworm', 'fhb', 'grasshopper', 'longhorn_beetle', 
            'oriental_fruit_fly', 'pesticides', 'plutella_xylostella', 
            'rice_planthopper', 'rice_stem_borer', 'rolled_leaf_borer'
        ]

        # Set a confidence threshold (e.g., 0.7 or 70%)
        confidence_threshold = 0.7

        if confidence.item() >= confidence_threshold:
            if 0 <= predicted_label < len(label_names):
                pest_name = label_names[predicted_label]
                description = pest_descriptions.get(pest_name, "No description available.")
                st.markdown(f"**Predicted Pest:** **{pest_name.upper()}**")
                st.write(description)
                st.markdown(f"[Read more about {pest_name} on Wikipedia](https://en.wikipedia.org/wiki/{pest_name})")
            else:
                st.write("Prediction is out of expected range. Please check the model output.")
        else:
            st.write("The model is not confident enough in its prediction for this image.")
    
# Error handling for potential issues
try:
    st.sidebar.info("image format jpeg, jpg or png")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
