import os

import chainlit as cl
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from llama_index.callbacks.base import CallbackManager
from llama_index.chat_engine import SimpleChatEngine
from llama_index.memory import ChatMemoryBuffer

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
SYSTTEM_MESSAGE = "You are a friendly chatbot living in 2023"


@cl.on_chat_start
async def on_start():
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    chat_engine = SimpleChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        prefix_messages=[],
    )

    cl.user_session.set("chat_engine", chat_engine)


@cl.on_message
async def on_message(message):
    chat_engine = cl.user_session.get("chat_engine")

    response = await cl.make_async(chat_engine.stream_chat)(message)

    response_message = cl.Message(content="")

    for token in response.response_gen:
        await response_message.stream_token(token=token)

    await response_message.send()
