import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# 1. Load environment variables
load_dotenv()
#groq_api_key = os.getenv("GROQ_API_KEY")

# 2. Load and split the document
loader = PyPDFLoader("generative-ai-harvard.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings and FAISS vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# 4. Set up retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

# 5. Load Groq LLM
llm = ChatGroq(
    #api_key=groq_api_key,
    model="llama-3.1-8b-instant"  # or any other supported model
)

# 6. Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an assistant with access to the following context:

{context}

Answer the question very strictly based on the above content in upto 5 to 6 lines.
If the answer is not in the content, respond with "I don't know."

Question: {question}
Answer:
"""
)

# 7. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# 8. Ask the bot
def ask_bot(question):
    response = qa_chain.invoke({"query": question})
    print("\nBot:", response["result"],"\n")

# Example
if __name__ == "__main__":
    print("\n                          Welcome to RAG-BASED-INFO-BOT!        ")
    print("RAG based Chatbot is ready to answer question's related to particular document! Ask any question. \n When you want to exit: Type 'exit or quit' to close app.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        ask_bot(query)
