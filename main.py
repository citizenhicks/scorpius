import os
import torch
import tyro
from dotenv import load_dotenv

from config import MODEL_CONFIGS, SamplerConfig
from weights import download_weights
from tokenizer import download_tokenizer, Tokenizer
from model import Model
from sampler import sample

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

def generate_stream(model, prompt):

    tokens = model.tokenize(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
    generated_tokens = []
    cur_pos = 0
    freqs_cis = model.precompute_freqs_cis()
    batch_size, seq_len = tokens.shape
    stop_words = torch.tensor([128001, 128008, 128009], device=model.getDevice(), dtype=torch.int64)
    cfg = SamplerConfig(model_size=model.getModelId())
    metrics_dict = {
            'logits_entropy': [],
            'logits_varentropy': [],
            'attn_entropy': [],
            'attn_varentropy': [],
            'angle': [],
            'magnitude': []
        }

    with torch.inference_mode():
        seq_len = tokens.shape[1]
        model.build_attn_mask(seq_len, cur_pos)
        logits, scores, attns_stats = model.forward(
                tokens, cur_pos, freqs_cis[:seq_len]
            )
        next_token = logits[:, -1].argmax(dim=-1)
        generated_tokens.append(next_token.item())
        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=1)
        cur_pos += 1
        yield model.detokenize(next_token)

        while cur_pos < model.params.max_seq_len:
            seq_len = tokens.shape[1]
            #model.build_attn_mask(seq_len, cur_pos)
            logits, scores, attns_stats = model.forward(
                tokens, cur_pos, freqs_cis[:seq_len]
            )

            next_token, metrics = sample(
                gen_tokens=tokens,
                logits=logits,
                attention_scores=scores,
                cfg=cfg,
                current_pos=cur_pos,
                device=model.getDevice()
            )
            for key in metrics_dict.keys():
                
                if key in metrics:
                    if key == 'magnitude':
                        metrics_dict[key].append(metrics[key])
                        continue
                    metrics_dict[key].append(metrics[key].item())

            generated_tokens.append(next_token.item())

            tokens = torch.cat([tokens, next_token], dim=1)
            cur_pos += 1

            yield model.detokenize(next_token)
            if torch.isin(next_token.to(model.getDevice()), stop_words).any():
                print("\n")
                print("entropy:", metrics_dict['logits_entropy'], "\nvarentropy:", metrics_dict['logits_varentropy'], "\nattention entropy:", metrics_dict['attn_entropy'], "\nattention varentropy:", metrics_dict['attn_varentropy'], "\n", "\nangle:", metrics_dict['angle'], "\nmagnitude:", metrics_dict['magnitude'])
                break

def generate_text(model, prompt):
    
    response = ""
    for token in generate_stream(model, prompt):
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
    print(f"Model {model_id} initialized and ready to use!\n")

    prompt = "Tell me a short story about love, keep it 3-4 sentence."
    
    generate_text(model=model, prompt=prompt)