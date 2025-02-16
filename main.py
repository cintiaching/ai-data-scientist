from agents.supervisor import get_ai_data_scientist

ai_data_scientist = get_ai_data_scientist()

# example usage of data_analyst_agent
result = ai_data_scientist.invoke({
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
result = ai_data_scientist.invoke({
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
