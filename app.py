
import os

import chainlit as cl

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


load_dotenv()


pdf_path = "Deepak_Garg AI Resume.pdf"

faiss_path = "faiss_index"


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


def create_vector_store():

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )

    vector_store.save_local(faiss_path)

    return vector_store


def load_vector_store():

    if os.path.exists(faiss_path):

        vector_store = FAISS.load_local(
            faiss_path,
            embedding,
            allow_dangerous_deserialization=True
        )

        return vector_store

    return create_vector_store()


vector_store = load_vector_store()


def retrieve_context(query):

    result = vector_store.similarity_search_with_score(
        query,
        k=8
    )

    context = "\n\n".join(
        document.page_content
        for document, score in result
    )

    return context


def generate_answer(query, context):

    prompt = f"""
You are a helpful resume assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:

"I could not find this information in the resume."

Do not invent information.

Context:

{context}

User Question:

{query}

Answer:
"""

    response = model.invoke(prompt)

    return response.text


@cl.on_chat_start
async def start():

    await cl.Message(
        content="""
# 🤖 Deepak's Resume Assistant

Welcome! I can answer questions about Deepak's resume.

You can ask me about:

- 🎓 Education
- 💻 Technical skills
- 🚀 Projects
- 🏢 Training and experience
- 📜 Certifications

**Ask your first question below!**
"""
    ).send()


@cl.on_message
async def main(message: cl.Message):

    query = message.content

    context = retrieve_context(query)

    answer = generate_answer(
        query,
        context
    )

    await cl.Message(
        content=answer
    ).send()