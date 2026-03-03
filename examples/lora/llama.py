#!/usr/bin/env python3
# pip3 install sentencepiece tiktoken blobfile
#import typeguard.importhook
#typeguard.importhook.install_import_hook('tinygrad')

from pathlib import Path
from typing import List, Optional
import argparse, json
from tinygrad import Tensor, Device, GlobalCounters, nn
from tinygrad.helpers import Context, Timing, Profiling, DEBUG, JIT, getenv, colored
from tinygrad.nn.state import safe_load, torch_load, load_state_dict, get_parameters
from extra.models.llama import Transformer, convert_from_huggingface, fix_bf16
from sentencepiece import SentencePieceProcessor
import sys
from extra.bench_log import BenchEvent, WallTimeEvent

MAX_CONTEXT = getenv("MAX_CONTEXT", 4096)

MODEL_PARAMS = {
  "7B": {
    "args": {"dim": 4096, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 11008},
    "files": 1,
  },
  "13B": {
    "args": {"dim": 5120, "n_heads": 40, "n_layers": 40, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 13824},
    "files": 2,
  },
  "70B": {
    "args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-05, "vocab_size": 32000, "hidden_dim": 28672},
    "files": 8,
  },
  "tokenizer": SentencePieceProcessor,
}

def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.startswith("tok_embeddings.") or name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
  return {name: convert(name) for name in {name: None for model in models for name in model}}

def load(fn:str):
  if fn.endswith('.index.json'):
    with open(fn) as fp: weight_map = json.load(fp)['weight_map']
    parts = {n: load(str(Path(fn).parent / Path(n).name)) for n in set(weight_map.values())}
    return {k: parts[n][k] for k, n in weight_map.items()}
  elif fn.endswith(".safetensors"):
    return safe_load(fn)
  else:
    return torch_load(fn)

class LLaMa:
  @staticmethod
  def build(model_path, tokenizer_path, model_size="7B", quantize=None, device=None):
    params = MODEL_PARAMS[model_size]
    tokenizer = MODEL_PARAMS['tokenizer'](model_file=str(tokenizer_path))
    assert tokenizer.vocab_size() == params["args"]["vocab_size"], f"{tokenizer.vocab_size()=} not equal to {params['args']['vocab_size']}"

    linear = nn.Linear
    model = Transformer(**params["args"], linear=linear, max_context=MAX_CONTEXT, jit=bool(JIT))

    with WallTimeEvent(BenchEvent.LOAD_WEIGHTS):
      if model_path.is_dir():
        weights = concat_weights([load(filename) for filename in [f"{model_path}/consolidated.{i:02d}.pth" for i in range(params["files"])]], device[0] if isinstance(device, tuple) else device)
      else:
        weights = load(str(model_path))
      if "model.embed_tokens.weight" in weights:
        weights = convert_from_huggingface(weights, params["args"]["n_layers"], params["args"]["n_heads"], params["args"].get("n_kv_heads", params["args"]["n_heads"]))

      weights = fix_bf16(weights)

      # prevent tracking model weights
      # this is a part of a larger problem with BUFFER UOps and gc in TRACK_MATCH_STATS=2
      with Context(BEAM=0, TRACK_MATCH_STATS=0):
        # quantize
        if quantize is not None:
          weights = linear.quantize(weights, device)
          for _,v in weights.items(): v.realize()

        # shard
        if isinstance(device, tuple):
          for k,v in nn.state.get_state_dict(model).items():
            if 'scale' in k: v.shard_(device, axis=None)  # from quantized
            elif '.attention.' in k:
              if getenv("SHARD_KVCACHE") and ('.wq.' in k or '.wk.' in k or '.wv.' in k): v.shard_(device, axis=0)
              else: v.shard_(device, axis=-1)
            elif '.feed_forward.w1.' in k: v.shard_(device, axis=0)
            elif '.feed_forward.w3.' in k: v.shard_(device, axis=0)
            elif '.feed_forward.' in k: v.shard_(device, axis=-1)
            elif 'tok_embeddings.weight' in k: v.shard_(device, axis=0)
            elif 'output.weight' in k: v.shard_(device, axis=-1)
            #elif k.endswith('.weight'): v.shard_(device, axis=-1)
            #elif 'norm.' in k: v.shard_(device, axis=-1)
            else: v.shard_(device, axis=None)

        # replace weights in model
        load_state_dict(model, weights, strict=False, consume=True)

    return LLaMa(model, tokenizer)

  def __init__(self, model, tokenizer):
    self.model = model
    self.tokenizer = tokenizer

if __name__ == "__main__":
  print(f"using {Device.DEFAULT} backend")

  parser = argparse.ArgumentParser(description="Run LLaMA in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--prompt", type=str, default=None, help="Phrase to start with. Without this, it goes into chatbot mode")
  parser.add_argument("--count", type=int, default=1000, help="Max number of tokens to generate")
  parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
  parser.add_argument("--timing", action="store_true", help="Print timing per token")
  parser.add_argument("--profile", action="store_true", help="Output profile data to out.prof")
  parser.add_argument("--size", type=str, default=None, help=f"""Size of model to use {list(MODEL_PARAMS.keys())}""")
  parser.add_argument("--quantize", type=str, default=None, help="Quantize the weights to int8 or nf4 in memory")
  parser.add_argument("--model", type=Path, default=None, required=True, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")
  parser.add_argument("--shard", type=int, default=1, help="number of devices to load the weights to")

  args = parser.parse_args()
  if args.size is None: args.size = "70B"
  if args.size not in MODEL_PARAMS: raise ValueError(f"Invalid model size: {args.size}")
  chatbot = args.prompt == None

  MODEL_PATH = args.model or Path(__file__).parents[2] / f"weights/LLaMA-2/{args.size}"
  TOKENIZER_PATH = (MODEL_PATH if MODEL_PATH.is_dir() else MODEL_PATH.parent) / "tokenizer.model"

  print(f"using LLaMA-2-{args.size} model")
  device = tuple(f"{Device.DEFAULT}:{i}" for i in range(args.shard)) if args.shard > 1 else Device.DEFAULT
  llama = LLaMa.build(MODEL_PATH, TOKENIZER_PATH, model_size=args.size, quantize=args.quantize, device=device)
  param_bytes = sum(x.uop.size * x.dtype.itemsize for x in get_parameters(llama.model))

  sys_prompt = ""
  start_pos = 0
  context = [llama.tokenizer.bos_id()] + llama.tokenizer.encode(sys_prompt)

  if args.prompt:
    context += llama.tokenizer.encode(args.prompt)

  curr_decoded = llama.tokenizer.decode(context)

  while 1:
    output: Tensor|None = None

    if chatbot:
      user_prompt = input("Prompt: ")
      context += llama.tokenizer.encode(user_prompt)

    for i in range(args.count):
      GlobalCounters.reset()

      if args.timing or args.profile: print("")
      st = GlobalCounters.time_sum_s
      next_tok = Tensor([context[start_pos:]], device=device) if output is None or len(context) > start_pos+1 else output.reshape(1, 1)

      with Profiling(enabled=args.profile):
        with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9/x:.2f} tok/s, {GlobalCounters.global_mem/x:.2f} GB/s, param {param_bytes/x:.2f} GB/s"):
          with WallTimeEvent(BenchEvent.STEP):
            with Timing("enqueue in ", on_exit=(lambda et: (f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on {Device.DEFAULT}" if DEBUG>=2 else "")+
                        f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"+
                        (f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s, param {param_bytes*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s" if DEBUG>=2 else "")) if DEBUG else None, enabled=args.timing):
              output = llama.model(next_tok, start_pos, args.temperature)
            tok = output.item()

      start_pos = len(context)
      context.append(tok)

      # TODO: this is a hack to deal with spaces. i think the decode is fast though, so who cares?
      decoded = llama.tokenizer.decode(context)
      sys.stdout.write(decoded[len(curr_decoded):])
      sys.stdout.flush()

      curr_decoded = decoded

      if tok == llama.tokenizer.eos_id(): break

    if not chatbot: break
