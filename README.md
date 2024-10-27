# PDF Querying and Summarization 🐦📄

## Overview
This project is a PDF Querying and Summarization app built using Streamlit. It allows users to upload PDF files, query the content for specific information, and generate concise summaries of the documents. The app utilizes advanced NLP models and embeddings to provide accurate responses and insights.

![Image 1](https://drive.google.com/uc?export=view&id=1sBULzxYzEVeDpxBTnhQCg5OuXCjsolmy)  ![Image 2](https://drive.google.com/uc?export=view&id=13qAMQdrUB5RYdHK-XUGyqSOHtq-8zb3B)

---

## 🚀 Features
- **PDF Querying**: Users can ask questions about the content of uploaded PDF files and receive relevant answers.
- **PDF Summarization**: Generates concise summaries of the PDF content to give users a quick overview.
- **Multiple File Uploads**: Supports uploading multiple PDF files for querying and summarization.
- **User-Friendly Interface**: Built with Streamlit for a smooth and interactive user experience.

---

## Project Structure
📂 PDF-Querying-and-Summarization  
📄 app.py # Main application file  
📄 requirements.txt # Project dependencies  
📄 README.md # Project documentation  

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- Required libraries: All dependencies are listed in `requirements.txt`.

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/PDF-Querying-and-Summarization.git
   cd PDF-Querying-and-Summarization
2. **Download the La-Mini Model**:
   - Download the model folder [here](https://drive.google.com/drive/folders/1_f8Ni-MGFwEn_aA8Mja5pjOpONFsRwZW?usp=sharing).
3. **Install dependencies:**:
   ```bash
   pip install -r requirements.txt
4. **Run the Application**:
   ```bash
   streamlit run app.py

---

##  🖥 Usage
- Open the application in your browser.
- Upload one or multiple PDF files.
- Use the PDF Querying feature to ask questions about the content.
- Summarize the PDF to obtain a concise overview of its contents.

--- 

##  📦 Dependencies
- To run this project, the following dependencies are needed:

1. numpy
2. pandas
3. matplotlib
4. streamlit
5. transformers
6. torch
7. pdfplumber
8. langchain
9. requests
10. Chroma
- All dependencies are listed in requirements.txt and can be installed in one command.

---

## 📊 Model Information
- The app uses a combination of NLP models for querying and summarization:
1. Querying: Utilizes Sentence Transformers for document embeddings and a pre-trained model for generating answers based on user queries.
2. Summarization: Uses a state-of-the-art summarization model to create concise summaries of the PDF content.

---

## 🔮 Future Enhancements
- Multi-format Support: Extend the application to support additional file formats like DOCX and TXT.
- Advanced Querying Techniques: Implement more sophisticated algorithms for improved querying capabilities.
- User Authentication: Add user login features to save query history and summaries.

---


## 🤝 Contributing
- If you wish to contribute, please fork the repository and submit a pull request. For major changes, open an issue to discuss the changes.
