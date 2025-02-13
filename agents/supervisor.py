from langgraph_supervisor import create_supervisor

from agents.llm import build_llm
from agents.data_analyst import create_data_analyst_agent

model = build_llm()

# agents
data_analyst_agent = create_data_analyst_agent()

# Create supervisor workflow
workflow = create_supervisor(
    [data_analyst_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a data analyst. "
        "For data analysis task, e.g. inquiry about data or data visualization, use data_analyst_agent. "
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What are the total sales generated in this fy?"
        }
    ]
})
print("ANSWER: ")
print(result["messages"][-1].content)
