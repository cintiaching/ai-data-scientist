import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.coder import python_repl_tool, Code
from agents.llm import build_llm

from dotenv import load_dotenv

load_dotenv(".env")

output_dir = os.getenv("OUTPUT_DIRECTORY", ".")
os.makedirs(output_dir, exist_ok=True)

model = build_llm()


@tool
def generate_python_pptx_code(user_input: str) -> str:
    """Generate python-pptx code given user input."""
    prompt = f"""You are an AI assistant specialized in creating PowerPoint presentations using the python-pptx library.
Extract key insights and generate relevant charts based on the past conversation. 
Finally, create a well-structured presentation that includes these charts and any necessary images, ensuring 
that the formatting is professional and visually appealing.
Afterward, save the presentation in pptx format in {output_dir} directory, 
give the file a relevant name.
"""
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                prompt + "Here is the user question:",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    code_gen_chain = code_gen_prompt | model.with_structured_output(Code)
    result = code_gen_chain.invoke({"messages": [("user", user_input)]})
    print("code generation result", result)
    return result.code


def create_slides_generator_agent():
    graph = create_react_agent(
        model,
        tools=[generate_python_pptx_code, python_repl_tool],
        name="slides_generator_agent"
    )
    return graph
