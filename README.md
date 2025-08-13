DEMO - https://drive.google.com/file/d/1l0v8Zp9y0kb7lWgBkxukgFcj9kW6LSdB/view?usp=drive_link
# Chat with Your PDF - A RAG-Powered Q&A Bot
This project is an AI-powered chatbot that allows you to have a conversation with any PDF document. It implements a Retrieval-Augmented Generation (RAG) pipeline to provide answers based solely on the content of the uploaded file, ensuring factual and contextually relevant responses.

The application is built with a user-friendly interface using Streamlit and features real-time streaming of answers for an interactive experience.

Project Architecture and Flow
This application follows a classic Retrieval-Augmented Generation (RAG) architecture. The process is straightforward and designed for accuracy and efficiency.

Document Ingestion & Preprocessing: When a user uploads a PDF, the application reads the text and splits it into small, semantically meaningful chunks. This is the "preprocessing" step.

Embedding & Indexing: Each text chunk is then converted into a numerical representation called an "embedding" using a local sentence-transformer model. These embeddings are stored in an in-memory ChromaDB vector database, creating an indexed, searchable library of the document's content.

Retrieval: When a user asks a question, the application embeds the question and uses the vector database to find the most relevant chunks from the original document.

Generation: The user's question and the retrieved chunks of text are combined into a detailed prompt. This prompt is then sent to the Google Gemini API, which generates a final, human-readable answer based only on the provided context.

Streaming Response: The answer from the AI is streamed back to the user token-by-token, creating a real-time "typing" effect in the chat interface.

Project Structure
You'll notice several folders in this repository. While some are empty, they represent a standard structure for a larger, more modular project.

rag.py: This is the core application file containing all the code.

/data: This folder is intended to hold the source PDF documents for the chatbot.

/src, /chunks, /vectordb, /notebooks: In a larger-scale project, these folders would be used to separate different parts of the code (e.g., data processing scripts, saved databases, experimental notebooks). For this self-contained Streamlit application, all processing is handled in memory, so these folders are not used.

Model and Embedding Choices Explained
Embedding Model (all-MiniLM-L6-v2): We use this popular model from Hugging Face to create the text embeddings. It runs entirely on your local machine, so this part of the process is 100% free and private. It was chosen because it offers an excellent balance of speed and performance for creating high-quality semantic representations of the text.

Language Model (gemini-1.5-flash): The original assignment specified using a locally-run open-source LLM like Mistral or Zephyr. While this approach was implemented, it resulted in an extremely slow user experience on standard hardware. The computational demand of these models led to long processing times and frequent errors related to local resource management.

To deliver a functional and high-performance application, the final version uses the Google Gemini API. This strategic decision was made for the following reasons:

Speed: It provides near-instantaneous responses, creating a much better user experience.

Quality: It leverages a state-of-the-art model for more accurate and coherent answers.

Accessibility: It is available through a generous free tier, making the application accessible without requiring high-end local hardware.

How to Run the Application
Follow these steps to set up the environment and run the chatbot. The preprocessing, embedding, and RAG pipeline are all handled automatically by the application.

1. Prerequisites
Python 3.8+

Git

2. Clone the Repository
git clone <your-repository-url>
cd <your-repository-name>

3. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# Create the environment
python -m venv .venv

# Activate the environment
# On Windows:
.\.venv\Scripts\Activate.ps1


4. Install Dependencies
Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

5. Get Your API Key
This project uses the Google Gemini API for generating answers.

Go to Google AI Studio to get your free API key.

Copy the key.

6. Run the Streamlit App
This single command starts the web server and opens the application in your browser. The streaming response is enabled by default.

streamlit run rag.py

Once the app is running, paste your Google Gemini API key into the sidebar, upload a PDF, and click "Process Document" to build the RAG pipeline. You can then start asking questions!

Sample Queries and Demo
Here are some example interactions with the chatbot using the provided AI Training Document.pdf.

(Here, you should insert screenshots of your application answering a few questions. This is a critical part of the README.)

Example Questions:

"What is this document about?"

"What are the fees and taxes for sellers?"

"What does the eBay Money Back Guarantee cover?"

Demo Video
Link to Demo Video
