from flask import Flask, request
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

import os

key = open('api-key', 'r').readline()
os.environ["OPENAI_API_KEY"] = key


def create_app():
    llm = ChatOllama(model='llama3', temperature=0)

    app = Flask(__name__)
    data = load_web_base()
    docs = make_docs_by_chunk_data(data)
    vector_store = make_vector_store(docs)

    @app.route('/chat')
    def ask():
        query = request.args["query"]
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.8}
        )

        parser = JsonOutputParser()

        prompt = PromptTemplate(
            template="Evaluate whether user queries and context are relevant.\n{format_instructions}\n",
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

        print(chain)

        return str(chain.invoke(query))

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
