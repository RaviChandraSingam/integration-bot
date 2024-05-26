import os

import streamlit as st
import vertexai
from vertexai.preview.language_models import TextGenerationModel



import uuid
import re

from typing import List, Tuple

from IPython.display import display, Image, Markdown

# from langchain.prompts import PromptTemplate
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain.storage import InMemoryStore

# from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser


PROJECT_ID = os.environ.get("GCP_PROJECT")  # Your Google Cloud Project ID
LOCATION = os.environ.get("GCP_REGION")  # Your Google Cloud Project Region
vertexai.init(project=PROJECT_ID, location=LOCATION)

from langchain_google_vertexai import (
    # VertexAI,
    ChatVertexAI,
    # VertexAIEmbeddings,
    VectorSearchVectorStore,
)

# retriever = any


# @st.cache_resource
# def get_model():

from langchain_google_vertexai import VertexAIEmbeddings
embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@003")

## load documents
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader('./AccountManagementSystem-AMS.txt')
docs = loader.load()

loader = TextLoader('./TransactionProcessingSystem-TPS.txt')
docs2 = loader.load()

loader = TextLoader('./PaymentGatewayIntegration-PGI.txt')
docs3 = loader.load()

loader = TextLoader('./CustomerInformationManagementSystem-CIMS.txt')
docs4 = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(docs)
# actualText = [d.page_content for d in docs]

from langchain_community.vectorstores import Chroma

vector_store = Chroma(persist_directory="./rag_content",
    collection_name="rag_data", embedding_function=embedding_model
)
retriever = vector_store.as_retriever()
retriever.vectorstore.add_documents(docs)
retriever.vectorstore.add_documents(docs2)
retriever.vectorstore.add_documents(docs3)
retriever.vectorstore.add_documents(docs4)
    # generation_model = TextGenerationModel.from_pretrained("text-bison@002")
    # return retriever

from langchain_core.prompts import ChatPromptTemplate
system = ("You name is Service Integrator AI."
          "You are software system architect tasking with providing system architecture advice.\n"
          "You will be given contextual text usually system or api documentation.\n"
          "Use this information only to provide advice and suggestions related to the user's question. \n"
          "Keep your answers grounded to contextual text only, do not hallucinate or deviate from user question. \n"
          # "Wrap code sections of your reply within ```. \n"
          # "Also include citations data in your reply. \n"
          "If provided context is not enough to answer user query then reply back with 'Sorry ; I do not have enough information to respond your queries.'")
# human = "Translate this sentence from English to French. I love programming."
# prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = [
        {
            "type": "text",
            "text": (
                # "You are software system architect tasking with providing system architecture advice.\n"
                # "You will be given contextual text usually system documentation.\n"
                # "Use this information to provide advice and suggestions related to the user's question. \n"
                f"User-provided question: {data_dict['question']}\n\n"
                "Context Text:\n"
                f"{formatted_texts}"
            ),
        }
    ]

    # Adding image(s) to the messages if present
    # if data_dict["context"]["images"]:
    #     for image in data_dict["context"]["images"]:
    #         messages.append(
    #             {
    #                 "type": "image_url",
    #                 "image_url": {"url": f"data:image/jpeg;base64,{image}"},
    #             }
    #         )
    # system = "I am a Developer and I want you to return information for the given question only from the provided as Context Text"
    return [("system", system),HumanMessage(content=messages)]

def split_image_text_types(docs):
    texts = []
    for doc in docs:
        # print(doc)
        texts.append(doc.page_content)
        # texts.append(doc.decode('utf-8'))
        # print(doc)
    return {"texts": texts}

# Create RAG chain
chain_multimodal_rag = (
    {
        "context": retriever | RunnableLambda(split_image_text_types),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(img_prompt_func)
    | ChatVertexAI(
        temperature=0, model_name="gemini-pro", max_output_tokens=1024
    )
)

def get_text_generation(prompt="", **parameters):

    # res = process_query(entityReportName=prompt)
    # generation_model = get_model()
    # response = generation_model.predict(prompt=prompt, **parameters)

    # retriever = get_model()
    result = chain_multimodal_rag.invoke(prompt)
# result
    return result.content

