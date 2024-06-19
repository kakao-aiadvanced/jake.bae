from flask import Flask, request
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from langchain_community.tools.tavily_search import TavilySearchResults

api_key = open('api-key', 'r').readline()
tavily_key = open('tavily-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = api_key
os.environ["TAVILY_API_KEY"] = tavily_key


def create_app():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    app = Flask(__name__)
    data = load_web_base()
    docs = make_docs_by_chunk_data(data)
    vector_store = make_vector_store(docs)

    @app.route('/chat')
    def ask():
        query = request.args["query"]
        documents = search_docs(query)
        state = {"query": query, "documents": documents}

        retry_count = 0
        (doc, retry_count) = check_relevance_and_retry(state, retry_count)

        if doc is None:
            search_content = web_search(query)
            return make_answer(query, search_content, False)

        if check_test_relevance("I like an apple", doc) is False:
            search_content = web_search(query)
            return make_answer(query, search_content, False)

        answer = make_answer(query, doc)

        if check_hallucination(answer, doc):
            answer = make_answer(query, doc, False)

        return str(answer)

    def search_docs(query):
        return vector_store.similarity_search(query, 3)

    def check_test_relevance(query, document):
        return check_relevance(query, document) is False

    def check_relevance_and_retry(state, retry_count):
        while retry_count < 3:
            if check_relevance(state["query"], state["documents"][retry_count]):
                return state["documents"][retry_count], retry_count
            retry_count += 1

        return None, retry_count

    def check_relevance(query, doc):
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""You must judge strictly. 
                     Evaluate whether user query and document are relevant. 
                     document" {document}
                     user query: {query}
                     return format is 'relevance: true' or 'relevance false'
                     {format_instructions}
                     """,
            partial_variables={"format_instructions": JsonOutputParser().get_format_instructions()}
        )

        chain = prompt | llm | JsonOutputParser()
        response = chain.invoke({"document": doc, "query": query})
        print(response)
        return "true" in str(response).lower()

    def make_answer(query, doc, hallucination=True):
        hallucination_remove = ""
        if hallucination is False:
            hallucination_remove = "don't make hallucination answer "

        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""You are instructor. 
                        Explain the user’s question in detail using the relevant document. 
                        And describe the source of the document, including the URL and title.
                        user's question: {query}
                        document: {document}\n
                     """ + hallucination_remove
        )

        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"document": doc, "query": query})
        print(response)
        return response

    def check_hallucination(answer, doc):
        prompt = PromptTemplate(
            input_variables=["query", "document"],
            template="""Check whether the answer contains hallucinated information by comparing it with the document.
                     answer: {answer}
                     document: {document}
                     return format is 'hallucination: true' or 'hallucination false'
                     {format_instructions}
                     """,
            partial_variables={"format_instructions": JsonOutputParser().get_format_instructions()}
        )

        chain = prompt | llm | JsonOutputParser()
        response = chain.invoke({"document": doc, "answer": answer})
        print(response)
        return "true" in str(response).lower()

    def web_search(query):
        web_search_tool = TavilySearchResults(k=1)
        return web_search_tool.invoke({"query": query})[0]

    return app


def load_web_base():
    loader = WebBaseLoader(["https://lilianweng.github.io/posts/2023-06-23-agent/",
                            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
                            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"])
    data = loader.load()
    return data


def make_docs_by_chunk_data(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.split_documents(data)
    return docs


def make_vector_store(docs):
    vector_store = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings(),
                                         persist_directory="./chroma_db", collection_name="lilianweng")
    return vector_store
