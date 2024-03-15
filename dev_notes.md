# 03/11/2024

# This project pretends to implement a Question Answering system from documents. The plan is to use LangChain and Hugging Face to achieve this.

# Useful links:
- [Hugging Face - Document Question Answering](https://huggingface.co/tasks/document-question-answering)
- [Hugging Face - Transformers Documentation - Document Question Answering](https://huggingface.co/docs/transformers/en/tasks/document_question_answering)
- [Hugging Face - Advanced RAG Tutorial](https://huggingface.co/learn/cookbook/en/advanced_rag)
- [Hugging Face Blog - Open Source LLMS as Agents](https://huggingface.co/blog/open-source-llms-as-agents)
- [LangChain - Question Answering Use Cases](https://python.langchain.com/docs/use_cases/question_answering) 

## **RAG should be implemented.** ([Simple RAG for GitHub issues using Hugging Face Zephyr and LangChain](https://huggingface.co/learn/cookbook/en/rag_zephyr_langchain))
RAG is a popular approach to address the issue of a powerful LLM not being aware of specific content due to said content not being in its training data, or hallucinating even when it has seen it before. **Such specific content may be *proprietary*, *sensitive*, or recent and updated often.**

RAG should be choosen over fine-tuning the LLM with the specific conten when the content often changes. Fine-tuning, when done repeatedly, leads to a "model shift" that changes the model's behavior.

RAG works by providing an LLM with additional context that is retrieved from relevant data so that it can generate a better-informed response.

### 14:20
Now I'm planning to ignore previous links and start using this one from LangChain that includes RAG and local models.
- [LangChain - Q&A with RAG Using local models](https://python.langchain.com/docs/use_cases/question_answering/local_retrieval_qa)

Some auxiliary links:
- [LangChain - Text Embedding with Hugging Face Hub](https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub)
- [LangChain - Chat Integration with Hugging Face](https://python.langchain.com/docs/integrations/chat/huggingface)
- [LangChain - Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders)


# 03/13/2024
Found two courses related to RAG:
- [Building and Evaluating Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/1/introduction)
- [Knowledge Graphs for RAG](https://learn.deeplearning.ai/courses/knowledge-graphs-rag/lesson/1/introduction)


# 03/15/2024
This short course from deeplearning.ai and LangChain explains how chat with your data using LangChain. 
- [LangChain Chat with Your Data](https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/1/introduction)

The plan is to adapt the course content to the project's needs.