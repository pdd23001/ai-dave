import os

from openai import OpenAI
from pydantic import BaseModel


"""Code for getting output from a model in a particular structured format. We use the pydantic library for this"""

client=OpenAI(base_url="https://integrate.api.nvidia.com/v1",
              api_key="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")


# --------------------------------------------------------------
# Step 1: Define the response format in a Pydantic model
# --------------------------------------------------------------

#creating class here

class CalendarEvent(BaseModel):
    name: str #key: datatype pairs. Model Identifies name, date and participants from the prompt if given that instruction like in this code.
    date: str
    participants: list[str]


# --------------------------------------------------------------
# Step 2: Call the model
# --------------------------------------------------------------

completion = client.beta.chat.completions.parse( #Have used client.beta function here***
    model="nvidia/llama-3.3-nemotron-super-49b-v1.5", #used different model (maybe pydantic not compatible with oss)
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Parth and Anand are going to the Cricket Tournament on Sunday the 25th of August",
        },
    ],
    response_format=CalendarEvent,
)

# --------------------------------------------------------------
# Step 3: Parse the response
# --------------------------------------------------------------

event = completion.choices[0].message.parsed
print(event.name)
print(event.date)
print(event.participants)
