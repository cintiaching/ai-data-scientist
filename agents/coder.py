import os
from typing import Annotated, TypedDict, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

from agents.agent import Agent
from agents.data_analyst import DataAnalystVanna

# This executes code locally, which can be unsafe
repl = PythonREPL()


class CoderState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class Code(BaseModel):
    """Schema for code solutions to questions about python."""
    prefix: str = Field(description="Description of the problem and approach")
    code: str = Field(description="Code block")


class CoderAgent(Agent):
    """Agent that code"""
    def __init__(self, vanna: DataAnalystVanna, model: BaseChatModel):
        system_prompt = """You are a coder agent, please use generate_python_code tool to generate code given user's intent.
And then use python_repl_tool to execute your code, and then return your result."""
        agent_name = "coder_agent"
        AgentState = CoderState

        super().__init__(model, agent_name, AgentState, system_prompt)
        self.vanna = vanna

    @staticmethod
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
    def generate_python_code(self, user_input: str) -> str:
        """Generate python code given user input."""
        ddl_list = self.vanna.get_related_ddl(user_input)
        doc_list = self.vanna.get_related_documentation(user_input)

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
5. Use print to show any result, e.g. model prediction.
6. You are not allowed to use the python-pptx module to create slides, response with "my role doesnt allow powerpoint 
creation, please use the slides_generator_agent" and return code=''.
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
        code_gen_chain = code_gen_prompt | self.model.with_structured_output(Code)
        result = code_gen_chain.invoke({"messages": [("user", user_input)]})
        print("code generation result", result)
        return result.code

    def get_tools_by_name(self):
        tools = [self.python_repl_tool, self.generate_python_code]
        self.model = self.model.bind_tools(tools)
        tools_by_name = {tool.name: tool for tool in tools}
        return tools_by_name
