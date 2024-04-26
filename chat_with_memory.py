from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


loader = PyPDFLoader("docs/caperucitaroja.pdf")
docs = loader.load()
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    # add_start_index=True
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

models = [
    "mistral-7b-openorca.gguf2.Q4_0.gguf", # https://gpt4all.io/models/gguf/mistral-7b-openorca.gguf2.Q4_0.gguf
    "mistral-7b-instruct-v0.1.Q4_0", # https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf
]

llm = GPT4All(
    model=f"models/{models[1]}",
    max_tokens=2048,
    temp=0.5,
    n_threads=8,
)

contextualize_q_system_prompt = """Se proporciona un historial de chat y la última pregunta \
    del usuario, la cual podría referirse al contexto del historial. Formula una pregunta independiente \
    que se pueda entender sin el historial. NO respondas la pregunta, solo reformúlala si es necesario, \
    de lo contrario, devuélvela tal cual. 
"""

contextualize_q_system_prompt = """Se proporciona un historial de chat y la última pregunta \
    del usuario, la cual podría referirse al contexto del historial. Si se refiere al contexto del historial, formula una pregunta independiente \
    que se pueda entender sin el historial. NO respondas la pregunta, solo reformúlala si es necesario, \
    de lo contrario, devuélvela tal cual. Si no se refiere al contexto del historial, devuélvela tal cual.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """Eres un asistente para responder preguntas. \
Utiliza la siguiente información del contexto para responder la pregunta. \
Si no sabes la respuesta, simplemente di que no la sabes. \
Intenta mantener la respuesta concisa y usar máximo tres oraciones. \
{context}
"""

qa_system_prompt = """Eres un asistente para responder preguntas e interacciones basicas. \
    Una interaccion basica se refiere a saludos, despedidas y preguntas como '¿Cómo estás?' o '¿Qué tal?'. \
    Si el usuario saluda o se despide, saluda o despidete. \
    De lo contrario utiliza la siguiente información del contexto para responder la pregunta. \
    Si no sabes la respuesta, simplemente di que no la sabes. \
    Intenta mantener la respuesta concisa y usar máximo tres oraciones. \
    {context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

## Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

try:
    while True:
        user_input = input("Usuario: ")
        answer = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": "abc123"}
            },
        )["answer"]
        print(answer)
except KeyboardInterrupt:
    print("\nExiting chat...")