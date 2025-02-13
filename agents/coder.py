import json
from typing import Annotated, TypedDict, Sequence

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from agents.llm import build_llm

model = build_llm()

# This executes code locally, which can be unsafe
repl = PythonREPL()


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
    system_prompt = """You are a python expert. Please help to generate a code to answer the question. 
Your response should ONLY be based on the given context and follow the response guidelines and format instructions. 
You can access to SQLite database if you need to, connect using
```python
import sqlite3

db_name = "data/sales-and-customer-database.db"

con = sqlite3.connect(db_name)
```
Do not delete or modify any data.
The tables within the database:
\n===Tables \nCREATE TABLE sales_data (invoice_no TEXT, customer_id TEXT, category TEXT, quantity INTEGER, price REAL, invoice_date TEXT, shopping_mall TEXT)\n\nCREATE TABLE customer_data (customer_id TEXT, gender TEXT, age REAL, payment_method TEXT)\n\n

\n===Additional Context \n\nThe invoice_date of sales_data is in dd-MM-yyyy format\n\nToday's date is 2025-02-13\n\nOur business defines financial year start with april to mar of each year\n

\n===Response Guidelines \n1. If the provided context is sufficient, please generate a valid python without any explanations for the question. \n2. If the provided context is insufficient, please explain why it can't be generated. \n4. Please use the most relevant table(s). \n5. Ensure that the output python is executable, and free of syntax errors. \n
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
    # {
    #     "code": result.code,
    #     "prefix": result.prefix
    # }


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
    # NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
    # code_agent = create_react_agent(
    #     model,
    #     tools=[generate_python_code, python_repl_tool],
    #     name="coder_agent",
    #     prompt="You are a coder agent, please use generate_python_code tool to generate code given user's intent"
    #            "And then use python_repl_tool to execute your code, and then return your result."
    # )
    # return code_agent
    # Define a new graph
    workflow = StateGraph(CoderState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `agent`.
    # This means that after `tools` is called, `agent` node is called next.
    workflow.add_edge("tools", "agent")

    # Now we can compile and visualize our graph
    graph = workflow.compile(name="coder_agent")
    return graph
