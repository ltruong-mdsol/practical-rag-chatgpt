import os
import openai
from llama_index.callbacks.base import CallbackManager

import chainlit as cl
from llama_index import LLMPredictor, ServiceContext
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv


from llama_index import VectorStoreIndex

from llama_index.memory import ChatMemoryBuffer
from llama_index import download_loader


load_dotenv("/Users/sangtnguyen/Coding/Personal/practical-rag/.env")
openai.api_key = os.environ.get("OPENAI_API_KEY")
SYSTTEM_MESSAGE = "You are a friendly chatbot living in 2023"


WikipediaReader = download_loader("WikipediaReader")
wikipedia_reader = WikipediaReader()

documents = wikipedia_reader.load_data(pages=["Cristiano_Ronaldo"])

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
    ),
)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    chunk_size=512,
)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(service_context=service_context)
