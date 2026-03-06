from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from src.document_explainer import validate_uploaded_image, extract_text_from_medical_image, explain_document_simple
import os


app = Flask(__name__)


load_dotenv(override=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip().strip('"').strip("'")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip().strip('"').strip("'")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in environment or .env file")
if PINECONE_API_KEY.startswith("your_") or not PINECONE_API_KEY.startswith("pcsk_"):
    raise ValueError("Invalid PINECONE_API_KEY format. Set your real Pinecone API key in .env")

if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in environment or .env file")
if GOOGLE_API_KEY.startswith("your_"):
    raise ValueError("Invalid GOOGLE_API_KEY format. Set your real Gemini API key in .env")


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/doc-explainer")
def doc_explainer():
    return render_template('doc_explainer.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


@app.route("/explain-document", methods=["POST"])
def explain_document():
    try:
        uploaded_file = request.files.get("document_image")
        user_question = request.form.get("question", "Explain this medical document in simple English").strip()

        if not user_question:
            user_question = "Explain this medical document in simple English"

        image_bytes, _ = validate_uploaded_image(uploaded_file)
        extracted_text = extract_text_from_medical_image(chatModel, image_bytes, user_question)
        explanation = explain_document_simple(chatModel, retriever, extracted_text, user_question)

        return {"ok": True, "explanation": explanation}
    except ValueError as value_error:
        return {"ok": False, "error": str(value_error)}, 400
    except Exception:
        return {"ok": False, "error": "Unable to process document right now. Please try again."}, 500



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
