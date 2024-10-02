# %%
import asyncio
import json
import os
import sys
from typing import Iterable
import tiktoken

import vertexai
from pydantic_core import ValidationError
from vertexai.generative_models import ChatSession, Content, GenerationResponse, GenerativeModel
from vertexai.preview import generative_models

from google.cloud import logging_v2

MAX_TOKENS = 8192
MODEL_NAME = "gemini-1.5-pro-002"

# Initialize the Cost Management client
client = logging_v2.Client()

def calc_costs(project_id, start_time, end_time):
    filter_str = f'''
    resource.type="aiplatform.googleapis.com/Endpoint"
    timestamp >= "{start_time}"
    timestamp <= "{end_time}"
    '''
    
    for entry in client.list_entries(project=project_id, filter_=filter_str):
        print(f"Timestamp: {entry.timestamp}")
        print(f"Resource: {entry.resource}")
        print(f"Payload: {entry.payload}")
        print("---")


rustsdk = open("/code/human3090/rustsdk.txt", "r", encoding="utf-8").read()
hyperdrive = open("/code/human3090/hyperdrive.txt", "r", encoding="utf-8").read()
hyperdrive_libraries = open("/code/human3090/hyperdrive_libraries.txt", "r", encoding="utf-8").read()
hyperdrive_summary = open("/code/human3090/hyperdrive_summary.txt", "r", encoding="utf-8").read()

# %%
def count_tokens(code_string, encoding_name="cl100k_base"):
    # sourcery skip: inline-immediately-returned-variable
    """Count the number of tokens in a given code string.

    Args:
        code_string: The code as a single string.
        encoding_name: The name of the encoding to use.  
                      "cl100k_base" is a good default for code.

    Returns:
        The number of tokens in the code string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(code_string))
    return num_tokens

token_count = count_tokens(hyperdrive)
print(f"Number of tokens in the hyperdrive codebase: {token_count:,.0f}")

# %%
def init_model(project, location) -> GenerativeModel:
    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name=MODEL_NAME)

# %%
def load_history() -> list[Content]:
    # create file if it doesn't exist
    if not os.path.exists("/code/human3090/session_history.json"):
        print("session history doesn't exist, creating it.")
        initialization_message = (
            f"You are an expert AI assistant to help explain and improve the codebase of Hyperdrive, the next-gen interest rate AMM."
            "You have access to the entire Solidity codebase in the format of Filename.sol:\n{CONTENT}"
            f"Here is the full codebase:\n\n{hyperdrive}\n\n{hyperdrive_libraries}"
        )
        session_history = [
            {
                "role": "user",
                "parts": [{"text": initialization_message}]
            },
            {
                "role": "model",
                "parts": [{"text": "Understood."}]
            },
        ]
        with open("/code/human3090/session_history.json", "w", encoding="utf-8") as file:
            json.dump(session_history, file, indent=4)
    # read in the content
    with open("/code/human3090/session_history.json", "r", encoding="utf-8") as file:
        session_history = json.load(file)
    return [Content.from_dict(content) for content in session_history]

# %%
def start_chat(model: GenerativeModel, history: list[Content] | None = None) -> ChatSession:
    return ChatSession(model=model, history=history)

# %%
def send_message(session: ChatSession, message: str, stream: bool = False, tokens: int = MAX_TOKENS) -> Iterable[GenerationResponse] | GenerationResponse:
    return session.send_message(
        content=message,
        generation_config=generative_models.GenerationConfig(
            max_output_tokens=tokens,
        ),
        stream=stream,
    )

# %%
async def respond_chunked(context, full_response, max_length=2000):
    # Split the full response into lines
    lines = full_response.split('\n')
    
    current_chunk = ""
    for line in lines:
        # If adding this line would exceed the max length, send the current chunk
        if len(current_chunk) + len(line) + 1 > max_length and current_chunk:
            await context.send(current_chunk)
            await asyncio.sleep(1)  # Add a small delay to avoid rate limiting
            current_chunk = ""
        
        # Add the line to the current chunk
        if current_chunk:
            current_chunk += '\n'
        current_chunk += line
    
    # Send any remaining content
    if current_chunk:
        await context.send(current_chunk)

# %%
def save_history(session: ChatSession):
    content_data = [obj.to_dict() for obj in session.history]
    json_data = json.dumps(content_data, indent=4)
    with open('/code/human3090/session_history.json', 'w', encoding="utf-8") as file:
        file.write(json_data)

# %%
def clear_output_interactive():
    """Clear output in an interactive environment."""
    from IPython.display import clear_output

    clear_output(wait=True)


def clear_output_non_interactive():
    """Clear output in a non-interactive environment."""
    if sys.platform == "win32":  # For Windows
        os.system("cls")
    else:  # For macOS and Linux
        os.system("clear")


def clear_output_robust():
    """Clear output irrespective of OS, and whether running interactively or non-interactively."""
    if "ipykernel" in sys.modules:
        clear_output_interactive()
    else:
        clear_output_non_interactive()

# %%
def parse_completion_stream(completion_stream, prompt):
    """Parse a completion stream and return the response."""
    response = []
    finished = False
    while not finished:
        try:
            next_item = next(completion_stream)
            assert isinstance(next_item, GenerationResponse)
            parts_list = next_item.candidates[0].content.parts
            if text := "".join(part.text for part in parts_list):
                response.append(text)
                clear_output_robust()
                print(prompt, flush=True)
                print("".join(response), flush=True)
            else:
                finished = True
        except (StopIteration, ValidationError):
            finished = True
    return "".join(response)