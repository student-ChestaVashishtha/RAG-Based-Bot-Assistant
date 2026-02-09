import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

PERSIST_DIR = "./chroma_db"

# Helper: format docs with page numbers (reduces hallucination + adds credibility)
def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"(Page {page}) {doc.page_content}")
    return "\n\n".join(formatted)


def process_pdf_and_chat(pdf_file, user_question, api_key):
    if not api_key:
        return "Please enter your Gemini API key."
    if not pdf_file:
        return "Please upload a PDF file."

    try:
        # 1. Load PDF
        loader = PyPDFLoader(pdf_file.name)
        documents = loader.load()

        # 2. Split text
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)

        # 3. Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # 4. Persistent Vector Store (KEY UPGRADE)
        if os.path.exists(PERSIST_DIR):
            vectordb = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
        else:
            vectordb = Chroma.from_documents(
                texts,
                embeddings,
                persist_directory=PERSIST_DIR
            )

        retriever = vectordb.as_retriever(search_kwargs={"k": 4})

        # 5. LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )

        # 6. Anti-hallucination Prompt (IMPORTANT)
        prompt = PromptTemplate.from_template(
            """
You are an assistant answering questions strictly from the given context.
Context:
{context}
Question:
{question}
Rules:
- Answer ONLY from the context.
- If the answer is not present, say:
  "I don't know based on the provided document."
- Mention page numbers when relevant.
"""
        )

        # 7. LCEL RAG Chain
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke(user_question)

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- UI ---------------- #

with gr.Blocks(theme="ocean") as demo:
    gr.Markdown("# 📄 PDF Intel-Link")
    gr.Markdown("Ask grounded questions from your PDF using Gemini + RAG.")

    with gr.Row():
        with gr.Column():
            key_input = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="Paste your API key here"
            )
            file_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a question about the PDF"
            )
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False
            )
            submit_btn = gr.Button("Analyze")

    submit_btn.click(
        fn=process_pdf_and_chat,
        inputs=[file_input, question_input, key_input],
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch()
