"""Experimental native MiniMax text prompt; no API service or model download.

python tools/minimax/chat.py --engine build/strata-minimax --model SHARD --prompt 'Hello'
"""
from __future__ import annotations
import argparse
import pathlib
import subprocess
import sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"tools"))
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine",required=True); p.add_argument("--model",type=pathlib.Path,required=True)
    p.add_argument("--prompt",required=True); p.add_argument("--gpu",default="0")
    p.add_argument("--context",type=int,default=4096); p.add_argument("--max-tokens",type=int,default=128)
    p.add_argument("--cache-mib",type=int,default=512)
    args=p.parse_args()
    meta=GGUFFile(args.model).metadata
    if meta.get("general.architecture")!="minimax-m3" or meta.get("tokenizer.ggml.pre")!="minimax-m2":
        raise ValueError("expected MiniMax-M3.1 GGUF metadata shard")
    template=meta.get("tokenizer.chat_template")
    if not template: raise ValueError("MiniMax GGUF has no chat template")
    # Render without a temporary file or changing the shared frontend's globals.
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=["jinja2.ext.loopcontrols"])
    def fail(message): raise ValueError(message)
    env.globals["raise_exception"]=fail
    text=env.from_string(template).render(messages=[{"role":"user","content":args.prompt}],tools=None,add_generation_prompt=True)
    tok=Tokenizer.from_gguf(args.model); ids=tok.encode(text,parse_special=True)
    cmd=[args.engine,"--model",str(args.model),"--tokens",",".join(map(str,ids)),"--gpu",args.gpu,
         "--context",str(args.context),"--n",str(args.max_tokens),"--cache-mib",str(args.cache_mib)]
    # Read the whole completion before decoding: Unicode byte tokens may span
    # several generated ids. Native engine diagnostics stay on stderr.
    result=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,check=True)
    output=[]
    for line in result.stdout.splitlines():
        if line.startswith("T "): output.append(int(line[2:]))
    print(tok.decode(output))
if __name__=="__main__": main()
