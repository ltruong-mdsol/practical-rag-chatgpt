import os

import chainlit as cl
import openai
from dotenv import load_dotenv
from llama_index import LLMPredictor
from llama_index.agent import ReActAgent
from llama_index.callbacks.base import CallbackManager
from llama_index.llms import OpenAI
from llama_index.memory import ChatMemoryBuffer
from llama_index.tools import QueryEngineTool, ToolMetadata
from query_rag import query_engine as quey_base_engine
from tools import code_interpreter_tool, show_image_tool
from vector_rag import query_engine as vector_base_engine

load_dotenv("/Users/sangtnguyen/Coding/Personal/practical-rag/.env")
openai.api_key = os.environ.get("OPENAI_API_KEY")

query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_base_engine,
        metadata=ToolMetadata(
            name="check_information",
            description="Provide information about Cristiano Ronaldo such as name, career,...",
        ),
    ),
    QueryEngineTool(
        query_engine=quey_base_engine,
        metadata=ToolMetadata(
            name="check_goal",
            description="Provide information about Ronaldo's goal in all career. Please give me raw user question",
        ),
    ),
    show_image_tool,
    code_interpreter_tool,
]


@cl.on_chat_start
async def on_start():
    llm = OpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

    agent = ReActAgent.from_tools(
        query_engine_tools,
        llm=llm,
        verbose=True,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        memory=memory,
    )

    print(agent._tools_dict)

    cl.user_session.set("chat_engine", agent)

    await cl.Message("![](./image.jpg)").send()


@cl.on_message
async def on_message(message):
    cl.user_session.set("image", None)
    image_path = None

    chat_engine = cl.user_session.get("chat_engine")

    response = await cl.make_async(chat_engine.chat)(message)

    image_path = cl.user_session.get("image")

    if image_path:
        elements = [
            cl.Image(
                name="Plot", display="inline", path=image_path, size="large"
            )
        ]
        print("Sending image")

        await cl.Message(content=response.response, elements=elements).send()

    else:
        response_message = cl.Message(content=response.response)

        await response_message.send()
