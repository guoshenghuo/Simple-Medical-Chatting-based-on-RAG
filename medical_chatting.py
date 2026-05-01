import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Set environment variable
os.environ['DASHSCOPE_API_KEY'] = 'sk-37746adfdf9641ef8fbe5d2821c69d16'

# 1. Load PDF document
def load_documents():
    pdf_path = "C:/Users/Bruce Xu/Desktop/2.pdf"  # Replace with your actual path
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages")
    return documents

# 2. Split document into chunks
def split_documents(docs):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split document into {len(chunks)} chunks")
    return chunks

# 3. Initialize embedding model and vector database
def init_vectorstore(chunks):
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    persist_directory = "./chroma"
    
    # Create or load vector database
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    db.persist()
    print("Vector database created and persisted")
    
    return Chroma(
        persist_directory=persist_directory, 
        embedding_function=embeddings
    )

# 4. Initialize conversation system
def init_conversation_system(vectordb):
    # Define prompt template (ALL ENGLISH)
    template = """
    You are a helpful assistant that provides temporary treatment plans for patients.
    Your first response should analyze the patient's symptoms and living habits to determine the possible condition.
    
    Example:
    User: "I have a headache and stayed up late studying recently."
    You: "Since you have been staying up late studying and have an irregular schedule, your condition is likely a tension headache."
    
    Example:
    User: "I have a headache and a stuffy nose."
    You: "You have both pain and nasal congestion. Based on these symptoms, you likely have a common respiratory condition, such as a cold."
    
    Your second response should provide non-pharmaceutical and pharmaceutical treatments.
    Name specific medications only if you are certain; otherwise, do not mention any.
    
    Example:
    User: "I have a headache and stayed up late studying recently."
    You: "Treatment plan:
    1. Immediate relief
    For tension headaches: gently squeeze the base of your middle fingers for 10 seconds to relieve pressure.
    2. Long-term prevention
    Avoid eye strain under fluorescent lights; use warm desk lamps to improve sleep quality.
    No medication is recommended as your condition is mild."
    
    Example:
    User: "I have a headache and a stuffy nose."
    You: "Treatment plan:
    1. Home remedies
    Drink herbal tea: 5g honeysuckle + 3g mint + 2g licorice.
    Get 15 minutes of sunlight daily to boost immunity.
    2. Environmental care
    Keep your room ventilated and clean to reduce allergens.
    3. Medication (if needed)
    You may take Ammonia Kahuang Min Capsules.
    Warnings: Avoid alcohol while taking; use with caution if you have high blood pressure."
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize LLM
    chatLLM = ChatTongyi(streaming=False)
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(return_messages=True)
    
    # Define retriever
    retriever = vectordb.as_retriever()
    
    # Function to get chat history
    def get_history(_):
        return memory.load_memory_variables({})["history"]
    
    # Build RAG chain
    rag_chain = (
        RunnableParallel({
            "context": retriever,
            "question": RunnablePassthrough(),
            "history": RunnableLambda(get_history)
        })
        | prompt
        | chatLLM
        | StrOutputParser()
    )
    
    return rag_chain, memory

# 5. Initialize the system
print("Initializing Medical Q&A System...")
documents = load_documents()
chunks = split_documents(documents)
vectordb = init_vectorstore(chunks)
rag_chain, memory = init_conversation_system(vectordb)
print("System initialization completed!")

# 6. Gradio response function
def respond(message, chat_history):
    response = rag_chain.invoke(message)
    memory.save_context({"input": message}, {"output": response})
    chat_history.append((message, response))
    return "", chat_history

# 7. Create Gradio UI (English)
with gr.Blocks(title="Medical Q&A Assistant") as demo:
    gr.Markdown("# 🩺 Medical Q&A Assistant")
    gr.Markdown("Describe your symptoms, and I will provide a preliminary diagnosis and treatment plan.")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Enter your symptoms")
    clear = gr.Button("Clear Conversation")
    
    def clear_memory():
        memory.clear()
        return []
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_memory, inputs=None, outputs=chatbot)

if __name__ == "__main__":
    demo.launch()