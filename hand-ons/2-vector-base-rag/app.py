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


@cl.on_chat_start
async def on_start():
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
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    index = VectorStoreIndex.from_documents(documents)

    chat_engine = index.as_chat_engine(
        chat_mode="react", memory=memory, service_context=service_context
    )

    print(chat_engine.tools)

    cl.user_session.set("chat_engine", chat_engine)


@cl.on_message
async def on_message(message):
    chat_engine = cl.user_session.get("chat_engine")

    response = await cl.make_async(chat_engine.stream_chat)(message)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    await response_message.send()
