import subprocess
import sys

import chainlit as cl
from llama_index.tools.function_tool import FunctionTool


def code_interpreter(code: str):
    """
    A function to execute python code, and return the stdout and stderr
    
    If you want make a plot. You must not display the chart directly using plt.show. You must save this into a file name 'image.jpg' then use load_image with input {'filename' : 'image.jpg'} in next step
    
    Don't use single quote ' such as Ronaldo's in your code. This will crash the app

    You should import any libraries that you wish to use. You have access to any libraries the user has installed.

    The code passed to this functuon is executed in isolation. It should be complete at the time it is passed to this function.

    You should interpret the output and errors returned from this function, and attempt to fix any problems.
    If you cannot fix the error, show the code to the user and ask for help

    It is not possible to return graphics or other complicated data from this function. If the user cannot see the output, save it to a file and tell the user.

    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return f"StdOut:\n{result.stdout}\nStdErr:\n{result.stderr}"


def show_image(filename: str):
    """
    plt.show alternative
    Only use this when you saved an jpg/png image already
    Show image to end user. This is the onlyway to show image/plot for user
    """
    cl.user_session.set("image", filename)

    print("Done, Image load to screen successfully. Ask user to check the image. Don't through anything")


code_interpreter_tool = FunctionTool.from_defaults(fn=code_interpreter)
show_image_tool = FunctionTool.from_defaults(fn=show_image)
