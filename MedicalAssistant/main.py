import streamlit as st
import os
import tempfile
from PIL import Image
import io
import fitz  # PyMuPDF for PDF processing
from fitz import Document  # Import Document class directly
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import base64
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from io import BytesIO



# 🔐 Set your Groq API key
load_dotenv()

# Initialize the Groq LLM
groq_llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY")  # Using the existing .env method
)

# Configure page
st.set_page_config(page_title="Medical Assistant", layout="wide")

# Initialize session state variables if they don't exist
if 'image_response' not in st.session_state:
    st.session_state.image_response = None
if 'pdf_response' not in st.session_state:
    st.session_state.pdf_response = None
if 'symptoms_response' not in st.session_state:
    st.session_state.symptoms_response = None
if 'disease_response' not in st.session_state:
    st.session_state.disease_response = None

# Function to encode images to base64
def encode_image(image_path_or_PIL_img):
    """Convert image to base64 format"""
    if isinstance(image_path_or_PIL_img, Image.Image):
        # This is a PIL image
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # This is an image path
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process images with Groq
def process_image_with_groq(image):
    """Process an image using Groq LLM to extract medical information"""
    try:
        # Convert image to base64
        b64_image = encode_image(image)

        # Create a comprehensive medical vision prompt
        vision_prompt = """You are a specialized medical image analysis AI.

        Examine the provided medical image thoroughly and report on:
        1. Any visible symptoms, conditions, abnormalities, or medical issues
        2. Relevant anatomical observations
        3. Potential preliminary diagnoses based solely on visual evidence
        4. Severity indicators if applicable

        Be descriptive, precise, and clinically focused. Avoid speculation beyond what is visually evident.
        Your analysis will be used by a medical AI system to provide information to patients."""

        # Create message with image content and the specialized prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": vision_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg;base64,{b64_image}"}
                    }
                ]
            }
        ]

        # Get response from model
        response = groq_llm.invoke(messages)
        return response.content
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return "Unable to process the image. Please try describing the image in text format instead."


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()
    os.unlink(tmp_path)  # Delete the temporary file

    return text


# Function to generate medical advice using Groq
def generate_medical_advice(input_text, input_type):
    """Generate medical advice using Groq LLM"""

    # Create a consistent system message for all input types
    system_message = """You are a professional medical assistant AI with extensive knowledge of medical conditions, symptoms, treatments, and healthcare protocols.

    The patient has provided information via {input_type} which is: {input_text}

    Analyze this input carefully and provide:
    1. Your assessment of the medical situation based on the provided information
    2. Potential conditions or diagnoses if appropriate (with confidence levels)
    3. Recommended next steps or treatments
    4. Clear indicators of when the patient should seek immediate medical attention
    5. Any relevant preventative measures or lifestyle recommendations

    Present your response in clear, compassionate language that a patient can understand without medical jargon.

    IMPORTANT: Always conclude with this disclaimer: "This information is for educational purposes only and does not constitute medical advice. Please consult with a qualified healthcare provider for diagnosis and treatment recommendations."
    """

    # Create messages for the chat model using a formatted system message
    formatted_system_message = system_message.format(
        input_type=input_type,
        input_text=input_text[:1000] + "..." if len(input_text) > 1000 else input_text  # Truncate if too long
    )

    messages = [
        SystemMessage(content=formatted_system_message),
        HumanMessage(content=input_text)
    ]

    # Get response from the model
    response = groq_llm.invoke(messages)

    return response.content

# Main UI

st.title("Medical Assistant")
st.markdown("""
This application helps you understand medical conditions based on:
- Symptoms you describe
- Medical images you upload
- Medical reports (PDF)
- Specific disease names
""")

# Create tabs for different input methods
tab1, tab2, tab3, tab4 = st.tabs(["Describe Symptoms", "Upload Image", "Upload Medical Report", "Specify Disease"])

with tab1:
    st.header("Describe Your Symptoms")
    symptoms = st.text_area("Please describe your symptoms in detail:", height=150)
    if st.button("Get Advice (Symptoms)"):
        if symptoms:
            with st.spinner("Analyzing symptoms..."):
                advice = generate_medical_advice(symptoms, "symptoms")
                st.session_state.symptoms_response = advice
        else:
            st.error("Please describe your symptoms first.")
    
    if st.session_state.symptoms_response:
        st.markdown("**Analysis Results:**")
        st.write(st.session_state.symptoms_response)

with tab2:
    st.header("Upload Medical Image")
    uploaded_image = st.file_uploader("Upload an image of the affected area or medical scan:", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Create a container for the image with a fixed width
        with st.container():
            # Display the image in a smaller size
            st.image(image, 
                    caption="Uploaded Image", 
                    use_container_width=False,  # Don't use full container width
                    width=300)  # Set a fixed width for the image

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Use Groq to analyze the image
                image_analysis = process_image_with_groq(image)
                # Then use Groq to generate medical advice based on the image analysis
                advice = generate_medical_advice(f"Based on image analysis: {image_analysis}", "image")
                st.session_state.image_response = advice
    
    if st.session_state.image_response:
        st.markdown("**Analysis Results:**")
        st.write(st.session_state.image_response)

with tab3:
    st.header("Upload Medical Report")
    uploaded_pdf = st.file_uploader("Upload your medical report:", type=["pdf"])
    if uploaded_pdf is not None:
        if st.button("Analyze Report"):
            with st.spinner("Analyzing report..."):
                report_text = extract_text_from_pdf(uploaded_pdf)
                advice = generate_medical_advice(report_text, "report")
                st.session_state.pdf_response = advice
    
    if st.session_state.pdf_response:
        st.markdown("**Analysis Results:**")
        st.write(st.session_state.pdf_response)

with tab4:
    st.header("Specify Disease")
    disease = st.text_input("Enter the name of the disease or condition:")
    if st.button("Get Information (Disease)"):
        if disease:
            with st.spinner(f"Getting information about {disease}..."):
                advice = generate_medical_advice(f"Provide comprehensive information about {disease}", "disease")
                st.session_state.disease_response = advice
        else:
            st.error("Please enter a disease or condition first.")
    
    if st.session_state.disease_response:
        st.markdown("**Analysis Results:**")
        st.write(st.session_state.disease_response)

# Add disclaimer at the bottom
st.markdown("---")
st.markdown("""**Important Disclaimer:** This application provides general information only and is not a substitute for professional medical advice, 
diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have 
regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read 
or received from this application.""")