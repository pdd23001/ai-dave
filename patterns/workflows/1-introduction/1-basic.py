import os 
from openai import OpenAI

#1: Create Client with API key

client=OpenAI( base_url="https://integrate.api.nvidia.com/v1",
              api_key="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")

#2: Create Completion

completion=client.chat.completions.create(model="openai/gpt-oss-120b", messages=[
                                          {"role":"system", "content":"You are a helpful assistant."},
                                          {"role":"user", "content":"Tell me when India gained independence."}])

#Use client.chat.completions.create function. Params are model="str" and messages=[{}{}] i.e. list of dicts with role system and role user.
# The content for system is you are a helpful assistant. For user it's the prompt


#3- Get response like below and print

response=completion.choices[0].message.content
print(response)