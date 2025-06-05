import gradio as gr
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tempfile
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from OCR import extract_text_from_pdf_ocr, extract_text_from_image


load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

main_prompt = ChatPromptTemplate.from_template(
    """You are a highly capable and informative assistant. Answer the question using only the information provided in the context. Avoid making assumptions or using outside knowledge. Prioritize clarity, depth, and helpfulness.

        <context>
        {context}
        </context>

        <chat_history>
        {chat_history}
        </chat_history>

        Current Task: Based on the above context and chat history, provide a detailed, accurate, and helpful response to the following question:

        {input}

        If the context does not provide enough information to answer fully, clearly state what is missing and what would be needed to answer the question properly."""
)

summary_prompt = ChatPromptTemplate.from_template(
    """You are an expert summarizer. Carefully review the documents below and produce a concise, accurate summary.

        Your summary should:
        - Focus on key points, main arguments, and significant findings.
        - Capture the essence of the content without omitting critical details.
        - Be written in clear, neutral language.
        - Stay under 300 words.

        Documents:
        {text}"""
)

VECTORS = None
SUMMARY = None
CHAT_HISTORY = []

def load_pdf_from_memory(file_obj):
    documents = []

    if hasattr(file_obj, "read"): 
        pdf_doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    elif isinstance(file_obj, str):
        pdf_doc = fitz.open(file_obj)
    else:
        raise ValueError("Unsupported file type received.")

    for page_num, page in enumerate(pdf_doc):
        text = page.get_text()
        if text.strip():
            documents.append(Document(page_content=text, metadata={"page": page_num + 1}))

    return documents

def generate_summary(docs):
    combined_text = "\n\n".join([doc.page_content for doc in docs[:8]])
    summarize_chain = summary_prompt | llm
    summary_response = summarize_chain.invoke({"text": combined_text})
    return summary_response.content

def handle_upload(file):
    global VECTORS, SUMMARY
    if file is None:
        return "No file provided."

    docs = []

    try:
        tmp_file_path = file
        pdf = fitz.open(tmp_file_path)

        for i, page in enumerate(pdf):
            text = ""
            if isinstance(file, str):
                tmp_file_path = file
            elif hasattr(file, "name"):
                tmp_file_path = file.name
            else:
                return "Unsupported file format."

            if not text or len(text) < 10:
                # Fallback to OCR
                try:
                    pix = page.get_pixmap(dpi=300)
                    image_path = f"{tmp_file_path}_page_{i}.png"
                    pix.save(image_path)

                    text = extract_text_from_image(image_path)
                    os.remove(image_path)
                    if not text.strip():
                        text = "[No text could be extracted from this page]"
                except Exception as e:
                    text = f"[OCR failed: {str(e)}]"

            docs.append(Document(page_content=text, metadata={"page": i + 1}))

        pdf.close()
        os.remove(tmp_file_path)
        if all(not doc.page_content.strip() or "[No text" in doc.page_content for doc in docs):
            try:
                docs = extract_text_from_pdf_ocr(tmp_file_path)
            except Exception as e:
                return f"Full OCR fallback failed: {str(e)}"

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        VECTORS = FAISS.from_documents(split_docs, embeddings)
        SUMMARY = generate_summary(split_docs)

        return SUMMARY

    except Exception as e:
        return f"Error processing document: {str(e)}"



def handle_question(chat_history, user_input):
    global VECTORS, CHAT_HISTORY
    if not VECTORS:
        chat_history = chat_history or []
        chat_history.append((user_input, "Please upload a document first."))
        CHAT_HISTORY = chat_history
        return chat_history

    document_chain = create_stuff_documents_chain(llm, main_prompt)
    retriever = VECTORS.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    formatted_history = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history]) if chat_history else ""

    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": formatted_history
    })

    answer = response['answer']
    chat_history = chat_history or []
    chat_history.append((user_input, answer))
    return chat_history

def reset_all():
    global VECTORS, SUMMARY, CHAT_HISTORY
    VECTORS = None
    SUMMARY = None
    CHAT_HISTORY = []
    return gr.update(value=None),"", [], ""

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Document Summarizer and Chat Q&A")

    with gr.Row():
        upload_box = gr.File(label="Upload PDF", file_types=[".pdf"], file_count="single")
        summary_output = gr.Textbox(label="Summary", lines=10)

    chatbot = gr.Chatbot(label="Chat with your documents")
    question_input = gr.Textbox(
        placeholder="Type your question here and press Enter...",
        show_label=False,
        lines=1
    )

    reset_btn = gr.Button("Reset")

    upload_box.change(fn=handle_upload, inputs=upload_box, outputs=summary_output)
    question_input.submit(fn=handle_question, inputs=[chatbot, question_input], outputs=chatbot)
    question_input.submit(lambda: "", None, question_input)
    reset_btn.click(fn=reset_all, outputs=[upload_box, summary_output, chatbot, question_input])

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
