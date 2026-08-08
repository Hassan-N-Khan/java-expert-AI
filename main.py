from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are a Java expert. Answer the following question in detail and provide code examples if necessary.

Here is the pdf notes to review from: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

if not os.path.exists(persist_directory):
    loader = PyPDFLoader("Java Programming.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
else:
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})