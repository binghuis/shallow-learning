import os
from typing import List
from openai import AzureOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment="gpt-35-turbo",
)


def completion(
    messages: str | List[ChatCompletionMessageParam],
    temperature=0,
):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, stream=True, temperature=temperature
    )
    ret = ""
    for chunk in stream:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            ret = ret + chunk.choices[0].delta.content
    return ret
