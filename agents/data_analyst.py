from vanna.openai import OpenAI_Chat
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from vanna.chromadb import ChromaDB_VectorStore

from agents.llm import get_azure_openai_client, build_llm
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

from agents.train_vanna import train

load_dotenv(".env")


class DataAnalystVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """powered by vanna"""

    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        azure_openai_client = get_azure_openai_client()
        OpenAI_Chat.__init__(self, client=azure_openai_client, config=config)


vn = DataAnalystVanna(config={"model": "gpt-4o-mini"})
vn.connect_to_sqlite("../data/sales-and-customer-database.db")
train(vn)
training_data = vn.get_training_data()
print("training_data", training_data)


# data analyst react agent
class DataAnalysisState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def answer_question_about_data(user_input: str) -> dict:
    """
    Call to get the answer about the data, and return a dictionary with the sql, sql execution result and answer
    :param user_input: (str) the question user ask
    :return: (dict) a dictionary containing the sql, execution_result, answer
    """
    sql = vn.generate_sql(user_input)
    sql_result = vn.run_sql(sql)
    answer = vn.generate_summary(user_input, sql_result)
    return {
        "sql": sql,
        "execution_result": str(sql_result),
        "answer": answer
    }


@tool
def visualize_data(user_input: str) -> dict:
    """
    Call to get data visualization plot about the data, and return a dictionary with the sql, sql execution result,
    plotly_code, and plotly_figure
    :param user_input: (str) the question user ask
    :return: (dict) a dictionary containing the sql, execution_result, plotly_code, and plotly_figure
    """
    sql = vn.generate_sql(user_input)
    df = vn.run_sql(sql)
    plotly_code = vn.generate_plotly_code(question=user_input, sql=sql,
                                          df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
    fig = vn.get_plotly_figure(plotly_code=plotly_code, df=df)
    return {
        "sql": sql,
        "execution_result": str(df),
        "plotly_code": plotly_code,
        "plotly_figure": fig
    }


tools = [answer_question_about_data, visualize_data]
model = build_llm()
model = model.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: DataAnalysisState):
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


# Define the node that calls the model
def call_model(
        state: DataAnalysisState,
        config: RunnableConfig,
):
    system_prompt = SystemMessage(
        "You are an data analyst, Always use one tool at a time."
        "For data analysis task / inquiry about the, use answer_question_about_data. "
        "For data visualization task, use visualize_data"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: DataAnalysisState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


def create_data_analyst_agent():
    # Define a new graph
    workflow = StateGraph(DataAnalysisState)

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
    graph = workflow.compile(name="data_analyst_agent")
    return graph
