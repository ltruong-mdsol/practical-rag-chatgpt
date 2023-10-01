import os
import openai
from llama_index.llms import ChatMessage, OpenAI

import chainlit as cl

from dotenv import load_dotenv

load_dotenv("/Users/sangtnguyen/Coding/Personal/practical-rag/.env")
openai.api_key = os.environ.get("OPENAI_API_KEY")


SYSTTEM_MESSAGE = "You are a friendly chatbot living in 2023"


@cl.on_chat_start
async def on_start():
    llm = OpenAI()
    cl.user_session.set("llm", llm)
    cl.user_session.set("messages", [
        ChatMessage(role="system", content=SYSTTEM_MESSAGE),
    ])
    await cl.Message("Welcome to RAG practical").send()


@cl.on_message
async def on_message(message):
    llm = cl.user_session.get("llm")
    messages = cl.user_session.get("messages")

    messages.append(
        ChatMessage(role="user", content=message)
    )

    response = llm.stream_chat(messages)

    response_message = cl.Message(content="")

    for token in response:
        await response_message.stream_token(token=token.delta)

    await response_message.send()

    messages.append(
        ChatMessage(role="assistant", content=response_message.content)
    )
    cl.user_session.set("messages", messages)
