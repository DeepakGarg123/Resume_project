
import os
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path = "Deepak_Garg AI Resume.pdf"
parser = StrOutputParser()
loader = PyPDFLoader(pdf_path)

documents = loader.load()

print("Total pages:", len(documents))

print(documents[0].metadata)

print(documents[0].page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)
chunks = splitter.split_documents(documents)

for i , chunk in enumerate(chunks[:10]):
    print(f"\n Chunk {i+1}:")
    print(chunk.page_content)
    print(chunk.metadata)

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
vector_store = FAISS.from_documents(
    documents=chunks , 
    embedding=embedding
)
vector_store.save_local('faiss index')
query = input("🤖 Ask questions about Deepak's Resume:")

result = vector_store.similarity_search_with_score(
    query,
    k=8
)
for i, (document, score) in enumerate(result):
    print(f"\n--- Retrieved Chunk {i+1} ---")
    print("Score:", score)
    print(document.page_content)

context = "\n\n".join(
    document.page_content
    for document, score in result
)

prompt = f"""
You are a helpful resume assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I could not find this information in the resume."

Context:
{context}

User Question:
{query}

Answer:
"""
model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash',
    temperature = 0
)
response = model.invoke(prompt)
print(response.text)

