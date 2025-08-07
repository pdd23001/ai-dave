import json
import os

"""Another Example of Function Calling Only just in a slightly different way"""

from openai import OpenAI
from pydantic import BaseModel, Field

client=OpenAI( base_url="https://integrate.api.nvidia.com/v1",
              api_key="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")

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# --------------------------------------------------------------
# Define the knowledge base retrieval tool
# --------------------------------------------------------------


def search_kb(question: str): #This returns the entire data to retrieve from. The Knowledge Base
    """
    Load the whole knowledge base from the JSON file.
    (This is a mock function for demonstration purposes, we don't search)
    """
    with open("kb.json", "r") as f:
        return json.load(f)


# --------------------------------------------------------------
# Step 1: Call model with search_kb tool defined
# --------------------------------------------------------------
#LLM sees the user prompt identifies the question which best matches from the knowldedge base which we have added in memory and then appends the answer there as well. Now the LLama parses this and gives the answer in arephrased way
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Get the answer to the user's question from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": { #These are the params
                    "question": {"type": "string"},
                },
                "required": ["question"], #required
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store." #prompt to give in system content

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What is the return policy?"},
]

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

completion.model_dump()

# --------------------------------------------------------------
# Step 3: Execute search_kb function
# --------------------------------------------------------------


def call_function(name, args):
    if name == "search_kb":
        return search_kb(**args)


for tool_call in completion.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------


class KBResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question.")
    source: int = Field(description="The record id of the answer.")

messages = [
    {"role": "system", "content": "You are a helpful assistant that answers questions from the knowledge base about our e-commerce store. Whenever giving answers rephrase slightly."},
    {"role": "user", "content": "What is the return policy?"},
]
completion_2 = client.beta.chat.completions.parse(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
    messages=messages,
    tools=tools,
    response_format=KBResponse,
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed
final_response.answer
final_response.source
print(final_response.answer)

# --------------------------------------------------------------
# Question that doesn't trigger the tool
# --------------------------------------------------------------

# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user", "content": "What is the weather in Tokyo?"},
# ]

# completion_3 = client.beta.chat.completions.parse(
#     model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
#     messages=messages,
#     tools=tools,
# )

# print(completion_3.choices[0].message.content)
