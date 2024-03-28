from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st



st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # file_name = uploaded_file.name
    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getvalue())
    # Read file as string
    loader = PyPDFLoader(temp_file)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(docs)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message(f"Tengo informacion sobre tu archivo llamado {uploaded_file.name}. Como te puedo ayudar?")

view_messages = st.expander("View the message contents in session state")

# Set up the LangChain, passing in Message History

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

gpt4all_model = "models/mistral-7b-openorca.gguf2.Q4_0.gguf"  # https://gpt4all.io/models/gguf/mistral-7b-openorca.gguf2.Q4_0.gguf
llm = GPT4All(
    model=gpt4all_model,
    max_tokens=2048,
    temp=0.5,
    n_threads=8,
)

prompt_template = PromptTemplate.from_template(
"""
    Eres un asistente para responder preguntas. 
    Utiliza los siguientes fragmentos de contexto recuperado para responder la pregunta. 
    Si no conoces la respuesta, simplemente di que no la sabes. 
    Usa máximo tres oraciones y mantén la respuesta concisa.
    Pregunta: {question}
    Contexto: {context}
    Respuesta:
"""
)

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
retriever = vectorstore.as_retriever(
search_type="similarity", 
search_kwargs={
        # Returns the top k documents
        "k": 6,
    }
)

rag_chain = (
    {"question": RunnablePassthrough(), "context": retriever | format_docs}
    | prompt_template
    | llm
    | StrOutputParser()
)
chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
    

""" MY CODE ---------------------------------------------------
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

def main():
    st.title("RAG Chatbot")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # file_name = uploaded_file.name
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getvalue())
        # Read file as string
        process_pdf(temp_file)
    
    gpt4all_model = "models/mistral-7b-openorca.gguf2.Q4_0.gguf"  # https://gpt4all.io/models/gguf/mistral-7b-openorca.gguf2.Q4_0.gguf
    llm = GPT4All(
        model=gpt4all_model,
        max_tokens=2048,
        temp=0.5,
        n_threads=8,
    )

    

def process_pdf(file_path):
    print(file_path)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    all_splits = text_splitter.split_documents(docs)
    
    return all_splits


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
response = rag_chain.invoke(question)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
    main()
"""