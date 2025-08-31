import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI  #  Use OpenAI LLM instead of Groq
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load .env file
load_dotenv()

# Fetch keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if "langchainvector" not in pc.list_indexes().names():
    pc.create_index(
        name="langchainvector",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

# Read PDF
pdfreader = PdfReader("budget_speech.pdf")
raw_text = ""
for page in pdfreader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

# Split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_text(raw_text)
print("Total Chunks:", len(texts))

# Use HuggingFace embeddings
embedding = OpenAIEmbeddings(model="text-embedding-3-small")  

# Load vectorstore
vectorstore = PineconeVectorStore.from_existing_index(
    index_name="langchainvector",
    embedding=embedding,
    namespace="langchain"
)

# Insert docs (if needed)
vectorstore.add_texts(texts[:50])
print("Inserted %i chunks." % len(texts[:50]))

# 🔑 Use GPT-4-mini instead of Groq
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",  #  GPT-4-mini
    temperature=0
)

first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()

    if query_text.lower() == "quit":
        break
    if query_text == "":
        continue

    first_question = False
    print("\nQUESTION: \"%s\"" % query_text)

    # Search relevant docs
    docs = vectorstore.similarity_search(query_text, k=10, namespace="langchain")

    # Combine context
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question: {query_text}
    """

    answer = llm.invoke(prompt)
    print("ANSWER:", answer.content, "\n")  # ✅ .content for OpenAI responses

    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc in docs:
        print(f"    \"{doc.page_content[:100]}...\"")
