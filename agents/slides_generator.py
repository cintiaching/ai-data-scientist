import json
import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import add_messages, StateGraph, END

from agents.coder import python_repl_tool, Code
from agents.llm.llm import build_llm

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


class SlidesGeneratorState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [python_repl_tool, generate_python_pptx_code]
model = model.bind_tools(tools, parallel_tool_calls=False)
tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: SlidesGeneratorState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}


def call_model(
        state: SlidesGeneratorState,
        config: RunnableConfig,
):
    system_prompt = SystemMessage(
        "You are a powerpoint slides generator agent, please use generate_python_code tool to creating PowerPoint "
        "presentations using the python-pptx library given user's intent"
        "And then use python_repl_tool to execute your code."
        f"Save the presentation in pptx format in {output_dir} directory."
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def should_continue(state: SlidesGeneratorState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def create_slides_generator_agent():
    workflow = StateGraph(SlidesGeneratorState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")
    graph = workflow.compile(name="slides_generator_agent")
    return graph
