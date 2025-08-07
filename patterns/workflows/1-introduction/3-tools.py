import json
import os

import requests
from openai import OpenAI
from pydantic import BaseModel, Field

client=OpenAI( base_url="https://integrate.api.nvidia.com/v1",
              api_key="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")

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# --------------------------------------------------------------
# Define the tool (function) that we want to call (We are accessing a free weather api using requests library)
# --------------------------------------------------------------


def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]


# --------------------------------------------------------------
# Step 1: Call model with get_weather tool defined
# --------------------------------------------------------------

tools = [ #format from OpenAI Docs
    {
        "type": "function", #type of tool - function
        "function": { #within the function there is name of our function and a description of what it does. There is also parameters.
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": { #type of params, propeties of params
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"], #This is the whether certain params are required or not.
                "additionalProperties": False, #no additional props
            },
            "strict": True, #mak
        },
    }
]

system_prompt = "You are a helpful weather assistant. Note that the unit of wind speed is m/s always when interacting with me." #system prompt to give to "system content" in chat completion"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in New York today?"},
]

completion = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=messages,
    tools=tools,
)

# --------------------------------------------------------------
# Step 2: Model decides to call function(s)
# --------------------------------------------------------------

completion.model_dump() #Note that LLM only gives the params we have to call the function in our script

# --------------------------------------------------------------
# Step 3: Execute get_weather function
# --------------------------------------------------------------


def call_function(name, args): #function for calling our function
    if name == "get_weather":
        return get_weather(**args)


for tool_call in completion.choices[0].message.tool_calls:  #going through tools calls and appending the tool_call.id and result of calling function to messages. This way LLM can print it for us
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(name, args)
    # print(result)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)} #kinda working with memory within the system
    )

# --------------------------------------------------------------
# Step 4: Supply result and call model again
# --------------------------------------------------------------

#format for weather response when using with Llama(chat.beta completion)
class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


completion_2 = client.beta.chat.completions.parse(
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5", #LLama for response since beta doesnt work with oss. We are printing the result function call that we got using this LLM 
    messages=messages,
    tools=tools,
    response_format=WeatherResponse,
)

# --------------------------------------------------------------
# Step 5: Check model response
# --------------------------------------------------------------

final_response = completion_2.choices[0].message.parsed
final_response.temperature
print(final_response.response)
