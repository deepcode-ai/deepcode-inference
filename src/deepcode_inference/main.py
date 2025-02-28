import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Type, Union
import fire
import torch
import torch.distributed as dist
from PIL import Image
from deepcode_common.protocol.instruct.messages import UserMessage, AssistantMessage, TextChunk, ImageChunk, ImageURLChunk
from deepcode_common.protocol.instruct.request import ChatCompletionRequest
from deepcode_common.tokens.tokenizers.base import Tokenizer
from deepcode_common.tokens.tokenizers.deepcode import DeepcodeTokenizer
from deepcode_common.tokens.tokenizers.tekken import Tekkenizer, SpecialTokenPolicy, is_tekken
from deepcode_inference.args import TransformerArgs
from deepcode_inference.generate import generate, generate_mamba
from deepcode_inference.mamba import Mamba
from deepcode_inference.transformer import Transformer

# Logging Configuration
logging.basicConfig(level=logging.DEBUG)

def is_torchrun() -> bool:
    """Check if we are running in a distributed setup."""
    return all(var in os.environ for var in ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"])

def load_tokenizer(model_path: Path) -> DeepcodeTokenizer:
    """Load the tokenizer from the given model path."""
    tokenizer_files = [f for f in os.listdir(model_path) if is_tekken(model_path / f) or f.endswith(".model")]
    if not tokenizer_files:
        raise FileNotFoundError(f"No tokenizer found in {model_path}. Please place a valid tokenizer file.")
    
    if len(tokenizer_files) > 1:
        raise ValueError(f"Multiple tokenizer files found: {', '.join(tokenizer_files)}. Ensure only one is present.")
    
    tokenizer_path = model_path / tokenizer_files[0]
    deepcode_tokenizer = DeepcodeTokenizer.from_file(str(tokenizer_path))
    
    if isinstance(deepcode_tokenizer.instruct_tokenizer.tokenizer, Tekkenizer):
        deepcode_tokenizer.instruct_tokenizer.tokenizer.special_token_policy = SpecialTokenPolicy.KEEP
    
    logging.info(f"Loaded tokenizer: {deepcode_tokenizer.instruct_tokenizer.__class__}")
    return deepcode_tokenizer

def get_model_cls(model_path: str) -> Union[Type[Mamba], Type[Transformer]]:
    """Return the model class (Mamba or Transformer) based on the model configuration."""
    with open(Path(model_path) / "params.json", "r") as f:
        args_dict = json.load(f)
    
    model_type = args_dict.get("model_type", "transformer")
    return {"mamba": Mamba, "transformer": Transformer}.get(model_type, Transformer)

def pad_and_convert_to_tensor(list_of_lists: List[List[int]], pad_id: int) -> List[List[int]]:
    """Pad the input lists to the maximum length."""
    max_len = max(len(lst) for lst in list_of_lists)
    return [[pad_id] * (max_len - len(lst)) + lst for lst in list_of_lists]

def get_multimodal_input() -> Tuple[UserMessage, bool]:
    """Prompt the user for multimodal input (text and images)."""
    chunks: List[Union[TextChunk, ImageChunk, ImageURLChunk]] = []
    
    text_input = input("Text prompt: ")
    if text_input:
        chunks.append(TextChunk(text=text_input))
    
    print("[You can input zero, one, or more images now.]")
    while True:
        image_input = input("Image path or URL (Leave empty to finish): ")
        if not image_input:
            break
        if Path(image_input).is_file():
            chunks.append(ImageChunk(image=Image.open(image_input)))
        elif image_input.startswith("http"):
            chunks.append(ImageURLChunk(image_url=image_input))
        else:
            logging.warning(f"Invalid image input: {image_input}")
    
    return UserMessage(content=chunks), not bool(chunks)

def interactive(model_path: str, max_tokens: int = 35, temperature: float = 0.7, num_pipeline_ranks: int = 1, instruct: bool = False, lora_path: Optional[str] = None) -> None:
    """Interactive mode for generating responses from the model."""
    should_print = not is_torchrun() or torch.distributed.get_rank() == 0
    num_pipeline_ranks = torch.distributed.get_world_size() if is_torchrun() else 1
    
    # Load tokenizer and model
    deepcode_tokenizer = load_tokenizer(Path(model_path))
    tokenizer = deepcode_tokenizer.instruct_tokenizer.tokenizer
    model_cls = get_model_cls(model_path)
    model = model_cls.from_folder(Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks)
    
    if lora_path:
        model.load_lora(Path(lora_path))

    # Interactive loop
    messages: List[Union[UserMessage, AssistantMessage]] = []
    while True:
        if should_print:
            if instruct:
                mm_input, finished = get_multimodal_input()
                if finished:
                    break
                messages.append(mm_input)
            else:
                user_input = input("Prompt: ")
                messages.append(UserMessage(content=user_input))

            chat_completion_request = ChatCompletionRequest(messages=messages)
            tokenized = deepcode_tokenizer.encode_chat_completion(chat_completion_request)
            tokens = tokenized.tokens
            images = tokenized.images
        else:
            tokens, images = [], []

        length_tensor = torch.tensor([len(tokens)], dtype=torch.int)
        if is_torchrun():
            dist.broadcast(length_tensor, src=0)

        generate_fn = generate if isinstance(model, Transformer) else generate_mamba
        generated_tokens, _ = generate_fn([tokens], model, [images], max_tokens=max_tokens, temperature=temperature, eos_id=tokenizer.eos_id)
        answer = tokenizer.decode(generated_tokens[0])

        if should_print:
            print(answer)
            print("=====================")

        messages.append(AssistantMessage(content=answer))

def demo(model_path: str, max_tokens: int = 35, temperature: float = 0, lora_path: Optional[str] = None) -> None:
    """Run a demo with predefined prompts."""
    should_print = not is_torchrun() or torch.distributed.get_rank() == 0
    num_pipeline_ranks = torch.distributed.get_world_size() if is_torchrun() else 1
    
    model_cls = get_model_cls(model_path)
    model = model_cls.from_folder(Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks)

    if lora_path:
        model.load_lora(Path(lora_path))

    deepcode_tokenizer = load_tokenizer(Path(model_path))
    tokenizer = deepcode_tokenizer.instruct_tokenizer.tokenizer

    prompts = ["This is a test", "This is another test", "Deepcode AI is great."]
    encoded_prompts = [tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts]

    generate_fn = generate if isinstance(model, Transformer) else generate_mamba
    generated_tokens, _logprobs = generate_fn(encoded_prompts, model, max_tokens=max_tokens, temperature=temperature, eos_id=tokenizer.eos_id)

    generated_words = [tokenizer.decode(prompt + token) for prompt, token in zip(prompts, generated_tokens)]

    if should_print:
        for word, logprob in zip(generated_words, _logprobs):
            print(word)
            logging.debug("Logprobs: %s", logprob)
            print("=====================")

if __name__ == "__main__":
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })
