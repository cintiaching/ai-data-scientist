from langgraph_supervisor import create_supervisor

from agents.coder import create_coder_agent
from agents.llm import build_llm
from agents.data_analyst import create_data_analyst_agent


def get_ai_data_scientist():
    model = build_llm()

    # agents
    data_analyst_agent = create_data_analyst_agent()
    coder_agent = create_coder_agent()

    # Create supervisor workflow
    workflow = create_supervisor(
        [data_analyst_agent, coder_agent],
        model=model,
        prompt=(
            "You are a team supervisor managing a data analyst. "
            "For data analysis task, e.g. inquiry about data or data visualization, use data_analyst_agent. "
            "For machine learning tasks or general coding task in python, use coder_agent."
        )
    )

    # Compile and run
    app = workflow.compile()
    return app
