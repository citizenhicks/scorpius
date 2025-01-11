import os
import torch
import tyro
from dotenv import load_dotenv

from config import MODEL_CONFIGS
from weights import download_weights
from tokenizer import download_tokenizer, Tokenizer
from model import Model

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
torch.set_float32_matmul_precision("high")

def generate_stream(model, tokens):
    generated_tokens = []
    cur_pos = 0
    freqs_cis = model.precompute_freqs_cis()
    batch_size, seq_len = tokens.shape

    while cur_pos < model.params.max_seq_len:
        seq_len = tokens.shape[1]
        model.build_attn_mask(seq_len, cur_pos)
        logits, scores, attns_stats = model.forward(
            tokens, cur_pos, freqs_cis[:seq_len]
        )

        next_token = logits[:, -1].argmax(dim=-1)
        generated_tokens.append(next_token.item())

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
        cur_pos += 1

        yield model.detokenize(next_token)

def generate_text(model, tokens):
    
    response = ""
    for token in generate_stream(model, tokens):
        print(token, end="", flush=True)
        response += token
    print()

if __name__ == "__main__":
    seed = 42
    load_dotenv(override=True)
    model_id = os.getenv("MODEL_ID", "1B")
    download_weights(model_id=model_id)
    path_to_tokenizer_weights = download_tokenizer()
    tokeniser = Tokenizer(path_to_tokenizer_weights)

    model_params = MODEL_CONFIGS[model_id]
    model = Model(model_id=model_id, params=model_params, tokeniser=tokeniser, device=DEVICE)
    print(f"Model {model_id} initialized and ready to use!")

    prompt = "Hello, how are you?"
    tokens = model.tokenize(prompt)
    

    tyro.cli(generate_text(model, tokens))