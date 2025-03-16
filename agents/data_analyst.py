from langchain_core.language_models import BaseChatModel
from vanna.openai import OpenAI_Chat
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from vanna.chromadb import ChromaDB_VectorStore

from agents.agent import Agent
from agents.llm.llm import get_llm_client


class DataAnalystVanna(ChromaDB_VectorStore, OpenAI_Chat):
    """powered by vanna"""
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        client = get_llm_client()
        OpenAI_Chat.__init__(self, client=client, config=config)


class DataAnalysisState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class DataAnalystAgent(Agent):
    """Agent that does data analysis using vanna"""
    def __init__(self, vanna: DataAnalystVanna, model: BaseChatModel):
        system_prompt = "You are an data analyst, Always use one tool at a time."
        "For data analysis task / inquiry about the, use answer_question_about_data. "
        "For data visualization task, use visualize_data"
        agent_name = "data_analyst_agent"
        AgentState = DataAnalysisState

        super().__init__(model, agent_name, AgentState, system_prompt)
        self.vanna = vanna

    @tool
    def answer_question_about_data(self, user_input: str) -> dict:
        """
        Call to get the answer about the data, and return a dictionary with the sql, sql execution result and answer
        :param user_input: (str) the question user ask
        :return: (dict) a dictionary containing the sql, execution_result, answer
        """
        try:
            sql = self.vanna.generate_sql(user_input, allow_llm_to_see_data=True)
            sql_result = self.vanna.run_sql(sql)
            answer = self.vanna.generate_summary(user_input, sql_result)
            return {
                "sql": sql,
                "execution_result": str(sql_result),
                "answer": answer,
            }
        except Exception as e:
            return {
                "sql": None,
                "execution_result": None,
                "answer": str(e),
            }

    @tool
    def visualize_data(self, user_input: str) -> dict:
        """
        Call to get data visualization plot about the data, and return a dictionary with the sql, sql execution result,
        plotly_code, and plotly_figure
        :param user_input: (str) the question user ask
        :return: (dict) a dictionary containing the sql, execution_result, plotly_code, and plotly_figure
        """
        try:
            sql = self.vanna.generate_sql(user_input)
            df = self.vanna.run_sql(sql)
            plotly_code = self.vanna.generate_plotly_code(question=user_input, sql=sql,
                                                  df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
            fig = self.vanna.get_plotly_figure(plotly_code=plotly_code, df=df)
            return {
                "sql": sql,
                "execution_result": str(df),
                "plotly_code": plotly_code,
                "plotly_figure": fig.to_dict()
            }
        except Exception as e:
            return {
                "sql": None,
                "execution_result": str(e),
                "plotly_code": None,
                "plotly_figure": None
            }

    def get_tools_by_name(self):
        tools = [self.answer_question_about_data, self.visualize_data]
        self.model = self.model.bind_tools(tools)
        tools_by_name = {tool.name: tool for tool in tools}
        return tools_by_name
