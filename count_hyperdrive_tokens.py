import tiktoken
import json

hyperdrive = open("/code/human3090/hyperdrive.txt", "r", encoding="utf-8").read()

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
with open("/code/human3090/session_history.json", "r", encoding="utf-8") as file:
    session_history = json.load(file)
total_context = 0
for message in session_history:
    for part in message["parts"]:
        if "text" in part:
            total_context += count_tokens(part["text"])

print(f"Total context tokens: {total_context:,.0f}")

# %%