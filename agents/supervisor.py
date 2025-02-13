from langgraph_supervisor import create_supervisor

from agents.coder import create_coder_agent
from agents.llm import build_llm
from agents.data_analyst import create_data_analyst_agent

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
# example usage of data_analyst_agent
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

# example usage of coder_agent
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What will our sales be next quarter? Please use Regression models to predict"
        }
    ]
})
print("ANSWER: ")
print(result["messages"][-1].content)

print("full answer")
print(result["messages"])
