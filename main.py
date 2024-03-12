from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

question = "Describe Q&A with retrieval"
docs = vectorstore.similarity_search(question)
print(len(docs))

print(docs[0])