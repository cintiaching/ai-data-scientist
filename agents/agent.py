import json
from abc import ABC, abstractmethod
from typing import TypedDict, Annotated, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END, add_messages


class State(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class Agent(ABC):
    def __init__(self, model: BaseChatModel, agent_name: str, system_prompt: str = None):
        self.model = model
        self.agent_name = agent_name
        self.system_prompt = system_prompt

    @abstractmethod
    def get_tools_by_name(self):
        pass

    def tool_node(self, state):
        tools_by_name = self.get_tools_by_name()
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

    def call_model(self, state):
        system_prompt = SystemMessage(self.system_prompt)
        response = self.model.invoke([system_prompt] + state["messages"])
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    @staticmethod
    def should_continue(state):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    def create_agent(self):
        workflow = StateGraph(State)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                # If `tools`, then we call the tool node.
                "continue": "tools",
                # Otherwise we finish.
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        graph = workflow.compile(name=self.agent_name)
        graph.get_graph().draw_mermaid()
        return graph
