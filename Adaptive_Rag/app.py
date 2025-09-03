
import os
from typing import List, Literal
from pydantic import BaseModel, Field
from pprint import pprint

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# -------------------------
# Load env
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not (OPENAI_API_KEY and PINECONE_API_KEY and TAVILY_API_KEY):
    raise EnvironmentError("Set OPENAI_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY")

# -------------------------
# Clients
# -------------------------
llm_router = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_gen = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

tavily_wrapper = TavilySearch(api_key=TAVILY_API_KEY)
web_search_tool = Tool(
    name="Tavily Search",
    func=tavily_wrapper.run,
    description="Search web for colleges, fees, placements, packages"
)

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "career-guidance-rag"

# -------------------------
# Example docs
# -------------------------
def load_example_docs():
    return [
        Document(page_content="""Career: Data Scientist
        Skills: Python, R, SQL, Statistics, Machine Learning, Deep Learning.
        Roadmap: Learn Python/SQL, stats, ML libs, projects, publish.
        Avg Package: ₹8–12 LPA freshers.
        Colleges: IIT Bombay, IIIT Hyderabad, ISI Kolkata.
        Future: High demand in healthcare, finance, robotics."""),
        Document(page_content="""Career: Web Developer
        Skills: HTML, CSS, JS, React, Node.js, MongoDB.
        Roadmap: Frontend → React → Backend → DBs → Projects.
        Avg Package: ₹4–8 LPA freshers.
        Colleges: IITs, NITs, Coding Bootcamps.
        Future: Evergreen demand in SaaS & Web3."""),
    ]

# -------------------------
# Build index (run once)
# -------------------------
def bootstrap_demo_index():
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    docs = load_example_docs()
    vs = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=INDEX_NAME,
        namespace="careers"
    )
    print(f"✅ Inserted {len(docs)} docs")

def get_retriever():
    return PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="careers"
    )

# -------------------------
# Router
# -------------------------
class RouteQuery(BaseModel):
    datasource: Literal["vectorstore", "web_search", "no_retrieval"]

router_system = """Route the user query:
- vectorstore: careers, skills, roadmaps
- web_search: colleges, fees, placements, events
- no_retrieval: trivial Qs"""

router_prompt = ChatPromptTemplate.from_messages(
    [("system", router_system), ("human", "{question}")]
)
question_router = router_prompt | llm_router.with_structured_output(RouteQuery)

# -------------------------
# Retrieval + rewrite
# -------------------------
def retrieve_from_vectorstore(q, k=5):
    retriever = get_retriever()
    return retriever.similarity_search(q, k=k)

rewrite_system = "Rewrite the question into a concise, retrieval-friendly query."
rewrite_prompt = ChatPromptTemplate.from_messages(
    [("system", rewrite_system), ("human", "{question}")]
)
question_rewriter = rewrite_prompt | llm_gen | StrOutputParser()

# -------------------------
# Graders
# -------------------------
class BinGrade(BaseModel):
    binary_score: str

rel_system = "Return yes if the doc is relevant to the question."
rel_prompt = ChatPromptTemplate.from_messages(
    [("system", rel_system), ("human", "Doc: {document}\n\nQ: {question}")]
)
rel_grader = rel_prompt | llm_gen.with_structured_output(BinGrade)

hall_system = "Return yes if answer is grounded in docs."
hall_prompt = ChatPromptTemplate.from_messages(
    [("system", hall_system), ("human", "Docs: {documents}\n\nAnswer: {generation}")]
)
hall_grader = hall_prompt | llm_gen.with_structured_output(BinGrade)

fit_system = "Return yes if the answer resolves the question."
fit_prompt = ChatPromptTemplate.from_messages(
    [("system", fit_system), ("human", "Q: {question}\n\nAnswer: {generation}")]
)
fit_grader = fit_prompt | llm_gen.with_structured_output(BinGrade)

# -------------------------
# RAG generation
# -------------------------
rag_system = """You are CareerGuideGPT. Use the context to answer helpfully.
- Be concise
- Suggest next steps
- Admit if info is missing"""
rag_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_system), ("human", "Context: {context}\n\nQ: {question}")]
)
rag_chain = rag_prompt | llm_gen | StrOutputParser()

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# -------------------------
# Web search wrapper
# -------------------------
def web_search_query(query: str) -> List[Document]:
    results = web_search_tool.invoke({"query": query})
    if isinstance(results, list):
        content = "\n\n".join([r.get("content", "") for r in results if isinstance(r, dict)])
    else:
        content = str(results)
    return [Document(page_content=content)]

# -------------------------
# Graph state + nodes
# -------------------------
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

def node_route(state):
    q = state["question"]
    print("Routing query...")
    src = question_router.invoke({"question": q})
    print(f"➡️ Routed to: {src.datasource}")
    if src.datasource == "web_search":
        return "web_search"
    elif src.datasource == "vectorstore":
        return "retrieve"
    return "generate"

def node_retrieve(state):
    q = state["question"]
    print("📚 Retrieving from VectorStore...")
    docs = retrieve_from_vectorstore(q, k=5)
    return {"question": q, "documents": docs}

def node_web_search(state):
    q = state["question"]
    print("🌐 Doing Web Search...")
    docs = web_search_query(q)
    return {"question": q, "documents": docs}

def node_grade_documents(state):
    q, docs = state["question"], state["documents"]
    filtered = []
    for d in docs:
        res = rel_grader.invoke({"document": d.page_content, "question": q})
        if res.binary_score == "yes":
            filtered.append(d)
    return {"question": q, "documents": filtered}

def node_transform_query(state):
    q = state["question"]
    better_q = question_rewriter.invoke({"question": q})
    docs = retrieve_from_vectorstore(better_q, k=5)
    return {"question": better_q, "documents": docs}

def node_generate(state):
    q, docs = state["question"], state.get("documents", [])
    print("📝 Generating final answer...")
    ctx = format_docs(docs)
    gen = rag_chain.invoke({"context": ctx, "question": q})
    return {"question": q, "documents": docs, "generation": gen}

def node_validate_generation(state):
    q, docs, gen = state["question"], state.get("documents", []), state.get("generation", "")
    docs_text = format_docs(docs)
    hall = hall_grader.invoke({"documents": docs_text, "generation": gen})
    if hall.binary_score != "yes":
        return "transform_query"
    fit = fit_grader.invoke({"question": q, "generation": gen})
    return "useful" if fit.binary_score == "yes" else "transform_query"

# -------------------------
# Graph wiring
# -------------------------
workflow = StateGraph(GraphState)
workflow.add_node("web_search", node_web_search)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("grade_documents", node_grade_documents)
workflow.add_node("transform_query", node_transform_query)
workflow.add_node("generate", node_generate)

workflow.add_conditional_edges(START, node_route, {
    "web_search": "web_search",
    "retrieve": "retrieve",
    "generate": "generate"
})
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")

def decide_from_grade(state):
    return "generate" if state.get("documents") else "transform_query"

workflow.add_conditional_edges("grade_documents", decide_from_grade, {
    "generate": "generate",
    "transform_query": "transform_query"
})
workflow.add_edge("transform_query", "generate")

def validate_edge(state):
    
    res = node_validate_generation(state)
    return res

workflow.add_conditional_edges("generate", validate_edge, {
    "useful": END,
    "transform_query": "transform_query"
})

app = workflow.compile()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # bootstrap index once
    bootstrap_demo_index()

    query = "What is the future scope of data scientists in India?"
    final_state=None
    for output in app.stream({"question": query}):
        for node, st in output.items():
            pprint(f"Node: {node}")
            final_state = st
    if final_state and "generation" in final_state:
        print("\nFinal Answer:\n", final_state["generation"])
    else:
        print("\n❌ No final answer generated.")
