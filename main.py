from agents.supervisor import get_ai_data_scientist

ai_data_scientist = get_ai_data_scientist()

# example usage of data_analyst_agent
thread_id = 1
result = ai_data_scientist.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What are the total sales generated in this fy?"
        }
    ]
},
    config={"thread_id": thread_id}
)
print("ANSWER: ")
print(result["messages"][-1].content)

# example usage of coder_agent
result = ai_data_scientist.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Use Regression models to predict the total sales next year"
        }
    ]
},
    config={"thread_id": thread_id}
)
print("ANSWER: ")
print(result["messages"][-1].content)

# example usage of coder_agent
result = ai_data_scientist.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Use Regression models to predict the total sales next year"
        }
    ]
},
    config={"thread_id": thread_id}
)
print("ANSWER: ")
print(result["messages"][-1].content)

# example usage of slides_generator_agent
result = ai_data_scientist.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Based on past conversation, create a slides to showcase the results"
        }
    ]
},
    config={"thread_id": thread_id}
)
print("ANSWER: ")
print(result["messages"][-1].content)


# example usage of agent collaboration
thread_id = 1
result = ai_data_scientist.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What were the top-performing products in the last quarter, "
                       "and can you use regression to predict the sales top-performing products in the coming quarter?"
                       "finally, create a ppt to present your result"
        }
    ]
},
    config={"thread_id": thread_id}
)
print("ANSWER: ")
print(result["messages"][-1].content)
