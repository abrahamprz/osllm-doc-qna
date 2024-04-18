from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from tqdm import tqdm

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


gpt4all_model = "models/mistral-7b-openorca.gguf2.Q4_0.gguf"  # https://gpt4all.io/models/gguf/mistral-7b-openorca.gguf2.Q4_0.gguf
llm = GPT4All(
    model=gpt4all_model,
    max_tokens=2048,
    temp=0.5,
    n_threads=8,
)

file_path = "/home/abraham/personal/osllm-doc-qna/docs/caperucitaroja.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
all_splits = text_splitter.split_documents(docs)


vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={
        # Returns the top k documents
        "k": 6,
    }
)


prompt_template = PromptTemplate.from_template(
"""
    # Eres un asistente para responder preguntas. 
    # Utiliza los siguientes fragmentos de contexto recuperado para responder la pregunta. 
    # Si no conoces la respuesta, simplemente di que no la sabes. 
    # Usa máximo tres oraciones y mantén la respuesta concisa.
    # Pregunta: {question}
    # Contexto: {context}
    # Respuesta:
"""
)
rag_chain = (
    {"question": RunnablePassthrough(), "context": retriever | format_docs}
    | prompt_template
    | llm
    | StrOutputParser()
)

question = "¿Quién es el personaje principal del cuento?"

# Add tqdm to show progress bar
with tqdm(total=1, desc="Processing") as pbar:
    response = rag_chain.invoke(question)
    pbar.update(1)

print(response)
