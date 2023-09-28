# Importing necessary libraries and modules
from threading import Thread  # For parallel execution
from typing import Iterator  # For type hinting the return of generator function

import torch  # Importing PyTorch library
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer  # Importing necessary classes from transformers library

# Specifying the model ID from Hugging Face Model Hub
model_id = 'codellama/CodeLlama-13b-Instruct-hf'#codellama/CodeLlama-13b-Instruct-hf

# Checking if CUDA (GPU) is available for model loading
if torch.cuda.is_available():
    # Loading configuration for the model
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1  # Setting pretraining_tp to 1 in the configuration
    
    # Loading the model with specific configurations if GPU is available
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,  # Setting the data type for model to float16
        load_in_4bit=True,  # Loading model in 4bit mode
        device_map='auto',  # Automatically mapping the model to available device
        use_safetensors=False,  # Not using safe tensors
    )
else:
    # If GPU is not available, model is set to None
    model = None

# Loading the tokenizer for the specified model
tokenizer = AutoTokenizer.from_pretrained(model_id)


# Function to construct the prompt string based on message, chat history, and system prompt
def get_prompt(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    do_strip = False  # Initial flag to control stripping of user input
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input  # Stripping user input based on flag
        do_strip = True  # Updating flag after first user input
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message  # Stripping message based on flag
    texts.append(f'{message} [/INST]')
    return ''.join(texts)  # Joining and returning the constructed prompt string


# Function to get the input token length of the constructed prompt
def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)  # Constructing the prompt
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']  # Tokenizing the prompt
    return input_ids.shape[-1]  # Returning the length of tokenized input


# Main function to run the model and generate responses
def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        top_k: int = 50) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)  # Constructing the prompt
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')  # Tokenizing the prompt and moving to GPU
    
    # Initializing TextIteratorStreamer for real-time output streaming
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    
    # Preparing keyword arguments for model generation
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    
    # Starting a new thread for model generation
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []  # List to hold the generated outputs
    for text in streamer:  # Iterating over the streamed texts
        outputs.append(text)  # Appending each text to outputs list
        yield ''.join(outputs)  # Yielding the concatenated string of outputs
