import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import GPT2LMHeadModel
model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

@dataclass
class GPT_config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="none")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Multi_atten(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_head = config.n_head #X (B, T, n_embd) -> (B, T, n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).reshape(1,1,config.block_size, config.block_size))

    def forward(self, x):
        B, T, n_embd = x.size()
        QKV = self.c_attn(x) #(B, T, 3*n_embd)
        q, k, v = torch.chunk(QKV, 3, dim=2)
        q = q.reshape(B,  T,self.config.n_head, -1).transpose(1,2)
        k = k.reshape(B,  T,self.config.n_head, -1).transpose(1,2)
        v = v.reshape(B, T, self.config.n_head, -1).transpose(1,2) #(B,  T, n_head,divied_emb)
        atten = q @ (k.transpose(-2,-1))/ math.sqrt(k.size(-1)) #(B, n_head, T, T)
        atten = atten.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))#(B, n_head, T, T)
        atten = torch.softmax(atten, dim = -1)
        logits = (atten @ v).transpose(1,2).contiguous() #(B, T, n_head,divied_emb)
        logits = logits.reshape(B, T, n_embd)
        logits = self.c_proj(logits)
        return logits

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = Multi_atten(config)# to be done
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x):#(B, T, n_embd)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
            ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, x): #x (B, T)
        B, T = x.shape
        position = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, T) # to get the batch dim (1, T)
        tokens = self.transformer.wpe(position) + self.transformer.wte(x)

        for block in self.transformer.h:
            tokens = block(tokens)
        tokens = self.transformer.ln_f(tokens)
        logits = self.lm_head(tokens)

        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # only dropout can be overridden see more notes below
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # we can override the dropout rate, if desired
        # create a from-scratch initialized minGPT model
        config = GPT_config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        #for k in sd_keys_hf:
        #    if any(k.endswith(w) for w in transposed):
        #        print(f"Transposing {k}")
        #    print(f"sd_hf[{k}].shape: {sd_hf[k].shape}")
        #    print(f"sd[{k}].shape: {sd[k].shape}")

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained("gpt2")
model.eval()
import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I am a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
print(tokens)
torch.manual_seed(42)
while tokens.size(1) < max_length:
    with torch.no_grad():
        result = model(tokens) #result [B, vocab_size]
        result = result[:, -1, :]
        #temperature = 0.7
        #result = result / temperature

        result = F.softmax(result, dim=-1)
        #print(sum(result))
        #print(result.shape)
        #print(result)
        value, index = torch.topk(result, 20, dim = -1) #index
        #print(value)
        #print(sum(value))
        new_tokens = torch.multinomial(value, 1)
        xcol = torch.gather(index, -1, new_tokens)
        tokens = torch.cat((tokens, xcol), dim=1)

print(tokens)
for i in range(5):
    print(enc.decode(tokens[i].tolist()))