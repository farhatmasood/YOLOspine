# This script counts the number of tokens in a text file using the tiktoken library.
# It is designed to work with the OpenAI GPT models, specifically for encoding text into tokens.


import tiktoken

# Replace 'your_text_file.txt' with your actual file name
file_path = 'yolov8m_20250418_181917_terminal_output.txt'

# Choose encoding based on your GPT model. Here 'cl100k_base' is suitable for GPT-4, GPT-3.5
encoding = tiktoken.get_encoding('cl100k_base')

# Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Count tokens
tokens = encoding.encode(text)
num_tokens = len(tokens)

print(f"Number of tokens: {num_tokens}")
