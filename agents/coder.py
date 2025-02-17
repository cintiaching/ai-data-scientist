import json
import os
from typing import Annotated, TypedDict, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

from agents.data_analyst import DataAnalystVanna
from agents.llm import build_llm

model = build_llm()

# This executes code locally, which can be unsafe
repl = PythonREPL()

vn = DataAnalystVanna(config={"model": "gpt-4o-mini", "client": "persistent", "path": "./vanna-db"})
vn.connect_to_sqlite(os.getenv("SQLITE_DATABASE_NAME", "data/sales-and-customer-database.db"))


class CoderState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class Code(BaseModel):
    """Schema for code solutions to questions about python."""
    prefix: str = Field(description="Description of the problem and approach")
    code: str = Field(description="Code block")


@tool
def python_repl_tool(
        code: Annotated[str, "the python code to execute."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
        print("code.code", code)
        print("code execution result", result)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str


@tool
def generate_python_code(user_input: str) -> str:
    """Generate python code given user input."""
    ddl_list = vn.get_related_ddl(user_input)
    doc_list = vn.get_related_documentation(user_input)

    system_prompt = f"""You are a python expert. Please help to generate a code to answer the question. 
Your response should ONLY be based on the given context and follow the response guidelines and format instructions. 
You can access to SQLite database if you need to, connect using
```python
import sqlite3

db_name = "{os.getenv("SQLITE_DATABASE_NAME", "data/sales-and-customer-database.db")}"

con = sqlite3.connect(db_name)
```
Close the connection at the end of the code. Do not delete or modify any data.
The tables within the database:
===Tables 
{"\n ".join(ddl_list)}

===Additional Context 
{"\n - ".join(doc_list)}

===Response Guidelines
1. If the provided context is sufficient, please generate a valid python without any explanations for the question.
2. If the provided context is insufficient, please explain why it can't be generated.
3. Please use the most relevant table(s). 
4. Ensure that the output python is executable, and free of syntax errors.
5. Do not print out raw data, or very long output.
    """
    code_gen_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt + "Here is the user question:",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    code_gen_chain = code_gen_prompt | model.with_structured_output(Code)
    result = code_gen_chain.invoke({"messages": [("user", user_input)]})
    print("code generation result", result)
    return result.code


tools = [python_repl_tool, generate_python_code]
model = model.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}


def tool_node(state: CoderState):
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
        state: CoderState,
        config: RunnableConfig,
):
    system_prompt = SystemMessage(
        "You are a coder agent, please use generate_python_code tool to generate code given user's intent"
        "And then use python_repl_tool to execute your code, and then return your result."
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: CoderState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def create_coder_agent():
    workflow = StateGraph(CoderState)
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
    graph = workflow.compile(name="coder_agent")
    return graph
