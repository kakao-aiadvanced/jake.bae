from flask import Flask, request
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain import LLMChain
from langchain_core.runnables import RunnablePassthrough
import os
import json

key = open('api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key


def create_app():
    llm = ChatOpenAI(temperature=0)

    app = Flask(__name__)
    data = load_web_base()
    docs = make_docs_by_chunk_data(data)
    vector_store = make_vector_store(docs)

    @app.route('/chat')
    def ask():
        query = request.args["query"]

        if not check_relevance(query):
            return "None"
        if not check_test_prompt():
            return "None"

        return "nice"

    def search_docs(query):
        return vector_store.similarity_search(query, 3)

    def check_test_prompt():
        return check_relevance("I like an apple") is False

    def check_relevance(query):
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.6}
        )
        parser = JsonOutputParser()
        prompt = PromptTemplate(
            input_variables=["query"],
            template="You must judge strictly. "
                     "Evaluate whether user queries and context are relevant.\n "
                     "user queries: {query}\n"
                     "return format is 'relevance: true or false'"
                     "{format_instructions}\n",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = {"context": retriever, "query": RunnablePassthrough()} | prompt | llm
        response = chain.invoke(query)
        response_json = json.loads(response.content)
        print(response_json)
        return "true" in str(response_json).lower()

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
