import os
from typing import Annotated, Sequence, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.graph import add_messages

from agents.agent import Agent
from agents.coder import python_repl_tool, Code

output_dir = os.getenv("OUTPUT_DIRECTORY", ".")
os.makedirs(output_dir, exist_ok=True)

class SlideGeneratorAgent(Agent):
    """Agent that code"""

    def __init__(self, model: BaseChatModel):
        system_prompt = "You are a powerpoint slides generator agent, please use generate_python_code tool to creating PowerPoint "
        "presentations using the python-pptx library given user's intent"
        "And then use python_repl_tool to execute your code."
        f"Save the presentation in pptx format in {output_dir} directory."
        agent_name = "slides_generator_agent"
        super().__init__(model, agent_name, system_prompt)

    @tool
    def generate_python_pptx_code(self, user_input: str) -> str:
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
        code_gen_chain = code_gen_prompt | self.model.with_structured_output(Code)
        result = code_gen_chain.invoke({"messages": [("user", user_input)]})
        print("code generation result", result)
        return result.code

    def get_tools_by_name(self):
        tools = [python_repl_tool, self.generate_python_pptx_code]
        self.model = self.model.bind_tools(tools)
        tools_by_name = {tool.name: tool for tool in tools}
        return tools_by_name
