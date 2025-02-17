from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor

from agents.coder import create_coder_agent
from agents.llm import build_llm
from agents.data_analyst import create_data_analyst_agent
from agents.slides_generator import create_slides_generator_agent


def get_ai_data_scientist():
    model = build_llm()
    # persistence
    checkpointer = InMemorySaver()

    # agents
    data_analyst_agent = create_data_analyst_agent()
    coder_agent = create_coder_agent()
    slides_generator_agent = create_slides_generator_agent()

    # Create supervisor workflow
    workflow = create_supervisor(
        [data_analyst_agent, coder_agent, slides_generator_agent],
        model=model,
        prompt=(
            "You are a team supervisor managing a data analyst, a coder and a slides generator. "
            "For data analysis task, e.g. inquiry about data or data visualization, use data_analyst_agent. "
            "For machine learning tasks or general coding task in python, use coder_agent. "
            "For creating powerpoint slides, use the slides_generator_agent. "
            "Think step by step and coordinate them to answer user's request. "
            "Give final response to the user based on the output from the agent(s)."
        )
    )

    # Compile and run
    app = workflow.compile(
        checkpointer=checkpointer,
        name="data_scientist"
    )
    return app
