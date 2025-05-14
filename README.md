🩺 MediScan: AI-Powered Medical Assistant
A comprehensive Streamlit application that provides AI-powered medical insights by analyzing your symptoms, medical images, reports, and disease queries — all powered by Groq LLM through LangChain.


🚀 Features

-🔍 Interactive symptom analysis with detailed guidance

-📸 Medical image interpretation and diagnosis support

-📄 PDF medical report processing and summarization

-🔬 Disease information lookup with evidence-based resources

-🔒 Secure API key handling with environment variables

-🏥 Professional medical disclaimers and ethical considerations



🛠️ Tech Stack


- Python 3.10+

- Streamlit

- LangChain & LangChain Groq

- Groq LLM 

- PyMuPDF (Fitz) for PDF processing

- PIL (Python Imaging Library) for image handling

- dotenv for secure API key management


🧠 How It Works

1. Users select from four different input methods: describing symptoms, uploading medical images, uploading PDF reports, or specifying a disease.

2. The input is processed according to its type (text analysis, image processing, PDF text extraction).

3. The processed data is sent to Groq's LLaMA-4 Scout model through LangChain for intelligent medical analysis.

4. Results are displayed in a user-friendly format with medical context and appropriate disclaimers.
