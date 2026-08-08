from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("Java Programming.pdf")
docs = loader.load()  # one Document per page

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",  # saves to disk so you don't re-embed every run
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # retrieve top 5 most similar chunks