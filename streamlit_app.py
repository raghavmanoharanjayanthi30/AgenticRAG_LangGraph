# Experiment with embedding models
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import os, yaml
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict

with open('chatgpt_api_credentials.yml') as f:
    credentials = yaml.safe_load(f)

os.environ['OPENAI_API_KEY']=credentials['openai_key']

openai_embed_model = OpenAIEmbeddings(model='text-embedding-3-small')

from langchain_chroma import Chroma
vector_store1 = Chroma(collection_name="openai_collection_1", embedding_function=openai_embed_model,
                      persist_directory="./chroma_db3")

retriever = vector_store1.as_retriever()

from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


#output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate two 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)
llm = ChatOpenAI(temperature=0)

# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)

# Run
multi_query_retriever = MultiQueryRetriever(
    retriever=vector_store1.as_retriever(), llm_chain=llm_chain #parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# 1.QUERY RETRIEVAL GRADER
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
# LLM for grading
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
# Prompt template for grading
SYS_PROMPT = """You are an expert grader assessing relevance of a retrieved document to a user question.
                Follow these instructions for grading:
                  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
                  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}
                     User question:
                     {question}
                  """),
    ]
)
# Build grader chain
doc_grader = (grade_prompt
                  |
              structured_llm_grader)

# 2.QUESTION REWRITER

SYS_PROMPT = """Act as a question re-writer and perform the following task:
                 - Convert the following input question to a better version that is optimized for web search.
                 - When re-writing, look at the input question and try to reason about the underlying semantic intent / meaning.
             """
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Here is the initial question:
                     {question}
                     Formulate an improved question.
                  """,
        ),
    ]
)
chatgpt = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# Create rephraser chain
question_rewriter = (re_write_prompt
                        |
                       chatgpt
                        |
                     StrOutputParser())

# FOR AGENTIC RAG
import os, yaml
with open('tavily_api_credentials.yml') as f:
    credentials = yaml.safe_load(f)
os.environ['TAVILY_API_KEY']=credentials['TAVILY_API_KEY']

from langchain_community.tools.tavily_search import TavilySearchResults
tv_search = TavilySearchResults(max_results=3, search_depth='advanced',max_tokens=10000)

# This text splitter is used to create the parent documents
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

vectorstore = Chroma(
    collection_name="full_documents", embedding_function=OpenAIEmbeddings()
)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
store = InMemoryStore()
p_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# QA RAG CHAIN
prompt1 = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.
            Question:
            {question}
            Context:
            {context}
            Answer:
         """
prompt_template1 = ChatPromptTemplate.from_template(prompt1)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    return "\n".join(f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history)


# QA RAG CHAIN
prompt2 = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context and conversation history to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.
            Conversation History:
            {history}
            Question:
            {question}
            Context:
            {context}
            Answer:
         """
prompt_template2 = ChatPromptTemplate.from_template(prompt2)

    
# create QA RAG chain with conversation history
# create QA RAG chain
qa_rag_chain = (
    {
        "context": (itemgetter('context') | RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
      |
    prompt_template1
      |
    chatgpt
      |
    StrOutputParser()
)

qa_rag_chain_with_history = (
    {
        "context": (itemgetter('context') | RunnableLambda(format_docs)),
        "question": itemgetter('question'),
        "history": itemgetter('history') | RunnableLambda(format_history)
    }
    | prompt_template2
    | chatgpt
    | StrOutputParser()
)

# BUILDING THE RAG AGENT
from typing import List
from typing_extensions import TypedDict
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]
    history: List[dict]

def retrieve_v2(state:GraphState):
    question = state['question']
    retrieved_docs = p_retriever.invoke(question)
    return {"documents": retrieved_docs, "question": question}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def grade_documents(state:GraphState):
    question = state['question']
    documents = state['documents']
    
    filtered_docs = []
    web_search_needed = 'No'
    if documents:
        for d in documents:
            score = doc_grader.invoke({'question':question, 'document': d.page_content})
            grade = score.binary_score
            if grade == 'yes':
                filtered_docs.append(d)
            else:
                web_search_needed = 'Yes'
    else:
        web_search_needed = 'Yes'
        
    return {"documents": filtered_docs, "question": question, 
            "web_search_needed": web_search_needed}
        
    
def rewrite_query(state:GraphState):
    question = state["question"]
    documents = state["documents"]
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

def web_search(state:GraphState):
    question = state['question']
    documents = state['documents']
    
    docs = tv_search.invoke(question)
    for i, d in enumerate(docs):
        if not isinstance(d, dict) or "content" not in d:
            continue
            # Handle the error, possibly skip this element or log it.

    web_results = "\n\n".join([d["content"] for d in docs if isinstance(d, dict) and "content" in d])
    #web_results = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}

def generate_answer(state:GraphState):
    question = state['question']
    documents = state['documents']
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, 
            "generation": generation}

def generate_answer_v2(state:GraphState):
    question = state['question']
    documents = state['documents']
    history = state['history']

    response = qa_rag_chain_with_history.invoke({"question": question, "context": documents, "history": history})
    history.append({"question": question, "answer": response})

    return {"documents": documents, "question": question, "generation": response, "history": history}

def should_generate(state:GraphState):
    web_search_needed = state['web_search_needed']
    
    if web_search_needed =='Yes':
        return 'web_search'
        #return 'rewrite_query'
    return 'generate_answer_v2'

# QA RAG CHAIN
prompt = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know the answer.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer and to the point answer with regard to the question.
            Question:
            {question}
            Context:
            {context}
            Answer:
         """
prompt_template = ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# create QA RAG chain
qa_rag_chain = (
    {
        "context": (itemgetter('context')
                        |
                    RunnableLambda(format_docs)),
        "question": itemgetter('question')
    }
      |
    prompt_template
      |
    chatgpt
      |
    StrOutputParser()
)

from langgraph.graph import END, StateGraph

agentic_rag = StateGraph(GraphState)
agentic_rag.add_node("retrieve_v2", retrieve_v2)
agentic_rag.add_node("grade_documents", grade_documents)
#agentic_rag.add_node("rewrite_query", rewrite_query)
agentic_rag.add_node("web_search", web_search)
agentic_rag.add_node("generate_answer_v2", generate_answer_v2)

agentic_rag.set_entry_point("retrieve_v2")
agentic_rag.add_edge("retrieve_v2", "grade_documents")

agentic_rag.add_conditional_edges(
    "grade_documents",
    should_generate,
    {"web_search": "web_search", "generate_answer_v2": "generate_answer_v2"},
)
#agentic_rag.add_edge("rewrite_query", "web_search")
agentic_rag.add_edge("web_search", "generate_answer_v2")
agentic_rag.add_edge("generate_answer_v2", END)
agentic_rag_compiled = agentic_rag.compile()

import streamlit as st
# Streamlit app
st.title("Document Q&A Agent with LangGraph")

# User input

count = 0
state1 = None

import json
import os

CACHE_FILE = "my_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            content = f.read()
            if content:  # Check if the file is not empty
                return json.loads(content)
            else:
                return {}  # Return an empty dictionary if the file is empty
    else:
        with open(CACHE_FILE, "w") as f:
            json.dump({}, f)

        return {}
    
def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

if 'count' not in st.session_state:
    st.session_state.count = 0
if 'state1' not in st.session_state:
    st.session_state.state1 = {'history': []}
if "messages" not in st.session_state:
    st.session_state.messages = []
# Load the cache from the file
if "cache" not in st.session_state:
    st.session_state.cache = load_cache()
        
#if "input_placeholder" not in st.session_state:
#    st.session_state.input_placeholder = st.empty()

# Always create the input box
# if st.session_state.count == 0:
#     query = st.session_state.input_placeholder.text_input("Ask a question about your documents:")
# else:
#     st.session_state.input_placeholder.empty() #clear previous input box.
#     st.session_state.input_placeholder = st.empty() #create new input box
#     query = st.session_state.input_placeholder.text_input("Ask another question:")

query = st.chat_input("Ask a question about your documents:")
import time

if query:
    query = query.strip()
    st.session_state.messages.append({'role': 'user', 'content': query})
    with st.chat_message(f'user'):
        st.markdown(query)

    with st.chat_message(f'assistant'):
        message_placeholder = st.empty()
        start_time = time.time()

        full_response = ""
        if query in st.session_state.cache:
            response = st.session_state.cache[query]
            end_time = time.time()
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.text(f"Response retrieved from cache. Time: {end_time - start_time:.4f} seconds")
        else:
            if st.session_state.count == 0:
                temp_state = {
                    'question': query,
                    'web_search_needed': 'No',
                    'documents': [],
                    'history': []
                }
            else:
                if len(st.session_state.state1['history']) > 5:
                    st.session_state.state1['history'] = st.session_state.state1['history'][-5:]
                temp_state = {
                    'question': query,
                    'web_search_needed': 'No',
                    'documents': [],
                    'history': st.session_state.state1['history']
                }
            st.session_state.state1 = agentic_rag_compiled.invoke(temp_state)

            response = st.session_state.state1['generation']
            end_time = time.time()
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.cache[query] = response
            st.text(f"Response generated. Time: {end_time - start_time:.4f} seconds")
        save_cache(st.session_state.cache)

    st.session_state.count += 1
#     with st.chat_message('documents'):
#         documents = st.session_state.state1['documents']
        
#         st.markdown("Documents used for generating answer: ")
#         for doc in documents:
#             st.markdown(doc.page_content)
#         #st.markdown(documents)
#         #st.markdown(response)
#         #st.session_state.messages.append({"role": "assistant", "content": response})
    
#     with st.chat_message('web_search'):
#         if st.session_state.state1['web_search_needed'].lower() == 'yes':
#             st.text("Web search was used to generate the answer.")
#         else:
#             st.text("Web search was not used to generate the answer.")

    

    # Clear and recreate input after answer
    #st.session_state.input_placeholder.empty()
    #st.session_state.input_placeholder = st.empty()
