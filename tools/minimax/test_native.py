"""Independent NumPy MiniMax oracle + tiny GGUF fixtures; no model download.

python tools/minimax/test_native.py --write-fixtures DIR
python tools/minimax/test_native.py --engine /path/to/strata-minimax [--gpu 0,1]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import struct
import subprocess
import tempfile
from contextlib import nullcontext
import numpy as np

CFG = dict(layers=3, width=256, heads=8, kv_heads=2, dim=32, rotary=16,
           dense_layers=1, dense_ff=512, experts=8, active=2, expert_ff=256,
           shared=1, index_heads=2, index_dim=32, top_blocks=2, block=4, local=1,
           context=64, eps=1e-6, theta=1e7, expert_scale=2.0)
TOKENS = [3, 4, 5, 8, 1, 6, 2, 9, 7, 0, 10, 11, 3, 13, 4, 15, 8, 1, 6]
KEYS = dict(layers="block_count", width="embedding_length", heads="attention.head_count",
            kv_heads="attention.head_count_kv", dim="attention.key_length", rotary="rope.dimension_count",
            dense_layers="leading_dense_block_count", dense_ff="feed_forward_length", experts="expert_count",
            active="expert_used_count", expert_ff="expert_feed_forward_length", shared="expert_shared_count",
            index_heads="attention.indexer.head_count", index_dim="attention.indexer.key_length",
            top_blocks="attention.indexer.top_k", block="attention.indexer.block_size",
            local="attention.indexer.local_blocks", context="context_length",
            eps="attention.layer_norm_rms_epsilon", theta="rope.freq_base", expert_scale="expert_weights_scale")

def string(s):
    b = s.encode(); return struct.pack("<Q", len(b)) + b

def encoded(a, mode):
    if a.ndim < 2 or mode == "f32": return 0, a.astype("<f4").tobytes(), a
    if mode == "nvfp4":
        # UE4M3 0x10 = 2^-5; E2M1 nibbles, halves split within each 16-value group.
        table=np.asarray([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
        codes=np.argmin(np.abs(a.reshape(-1,4,16)[...,None]*32-table),axis=-1).astype(np.uint8)
        pairs=codes[...,:8] | (codes[...,8:]<<4)
        raw=np.concatenate([np.full((len(codes),4),0x10,np.uint8),pairs.reshape(-1,32)],axis=-1)
        return 40,raw.tobytes(),(table[codes]/32).reshape(a.shape)
    if mode == "bf16":
        u = (a.view(np.uint32) >> 16).astype("<u2")
        return 30, u.tobytes(), (u.astype(np.uint32) << 16).view(np.float32)
    # Independent GGML Q8_0 encoder: fp16 scale then 32 signed codes.
    blocks = a.reshape(-1, 32)
    scale = np.max(np.abs(blocks), axis=1) / 127
    codes = np.rint(blocks / scale[:, None]).clip(-127, 127).astype(np.int8)
    rounded = scale.astype("<f2")
    payload = b"".join(rounded[i:i+1].tobytes() + codes[i].tobytes() for i in range(len(blocks)))
    values = (rounded.astype(np.float32)[:, None] * codes).reshape(a.shape)
    return 8, payload, values

def fixture(path, mode="f32"):
    rng = np.random.default_rng(713)
    c = CFG; N=c["width"]; D=c["dim"]; Q=D*c["heads"]; K=D*c["kv_heads"]
    w = {}
    def mat(name, *shape):
        w[name] = (rng.normal(size=shape) * (0.45 / np.sqrt(shape[-1]))).astype(np.float32)
    def norm(name, dim): w[name] = (1 + rng.normal(size=dim) * .04).astype(np.float32)
    mat("token_embd.weight",32,N); mat("output.weight",32,N); norm("output_norm.weight",N)
    for l in range(c["layers"]):
        p=f"blk.{l}."
        norm(p+"attn_norm.weight",N); norm(p+"ffn_norm.weight",N)
        norm(p+"attn_q_norm.weight",D); norm(p+"attn_k_norm.weight",D)
        for name,shape in [("attn_q",(Q,N)),("attn_k",(K,N)),("attn_v",(K,N)),("attn_output",(N,Q))]: mat(p+name+".weight",*shape)
        if l < c["dense_layers"]:
            F=c["dense_ff"]
            mat(p+"ffn_gate.weight",F,N); mat(p+"ffn_up.weight",F,N); mat(p+"ffn_down.weight",N,F)
        else:
            F=c["expert_ff"]; E=c["experts"]
            mat(p+"ffn_gate_inp.weight",E,N)
            w[p+"exp_probs_b.bias"] = (rng.normal(size=E)*.2).astype(np.float32)
            mat(p+"ffn_gate_exps.weight",E,F,N); mat(p+"ffn_up_exps.weight",E,F,N); mat(p+"ffn_down_exps.weight",E,N,F)
            mat(p+"ffn_gate_shexp.weight",F,N); mat(p+"ffn_up_shexp.weight",F,N); mat(p+"ffn_down_shexp.weight",N,F)
        mat(p+"indexer.q_proj.weight",c["index_heads"]*c["index_dim"],N)
        mat(p+"indexer.k_proj.weight",c["index_dim"],N)
        norm(p+"indexer.q_norm.weight",c["index_dim"]); norm(p+"indexer.k_norm.weight",c["index_dim"])
    meta = {"general.architecture":"minimax-m3", "general.name":"MiniMax-M3.1-Flash tiny fixture"}
    meta.update({"minimax-m3."+KEYS[k]:v for k,v in c.items()})
    meta.update({"minimax-m3.attention.value_length":D,"minimax-m3.expert_gating_func":2,
                 "minimax-m3.expert_weights_norm":True})
    header=bytearray(struct.pack("<IIQQ",0x46554747,3,len(w),len(meta)))
    for k,v in meta.items():
        header += string(k)
        if isinstance(v,str): header += struct.pack("<I",8)+string(v)
        elif isinstance(v,bool): header += struct.pack("<I?",7,v)
        elif isinstance(v,float): header += struct.pack("<If",6,v)
        else: header += struct.pack("<II",4,v)
    data=bytearray(); expected={}
    for name,a in w.items():
        if mode=="nvfp4" and name.endswith("ffn_down_exps.weight"):
            scales=np.linspace(.5,1.5,c["experts"],dtype=np.float32)
            ty,payload,values=encoded(a/scales[:,None,None],"nvfp4")
            values=values*scales[:,None,None]
        else: ty,payload,values=encoded(a,"q8" if mode=="nvfp4" else mode)
        expected[name]=values
        header+=string(name)+struct.pack("<I",a.ndim)+struct.pack("<"+"Q"*a.ndim,*reversed(a.shape))+struct.pack("<IQ",ty,len(data))
        data+=payload; data+=b"\0"*((-len(data))%32)
    if mode=="nvfp4":
        # Add independent per-expert scales to the tensor directory and count.
        extras=[name[:-7]+".scale" for name in w if name.endswith("ffn_down_exps.weight")]
        struct.pack_into("<Q",header,8,len(w)+len(extras))
        for name in extras:
            header+=string(name)+struct.pack("<IQIQ",1,c["experts"],0,len(data))
            data+=np.linspace(.5,1.5,c["experts"],dtype="<f4").tobytes(); data+=b"\0"*((-len(data))%32)
    header+=b"\0"*((-len(header))%32)
    path.write_bytes(header+data)
    return expected

def rms(x,w): return x / np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+CFG["eps"]) * w
def rotary(x,pos):
    r=CFG["rotary"]; half=r//2
    a=pos/(CFG["theta"]**(np.arange(half,dtype=np.float32)*2/r))
    out=x.copy(); z=x[...,:half]+1j*x[...,half:r]; z=z*np.exp(1j*a)
    out[...,:half]=z.real; out[...,half:r]=z.imag
    return out
def swiglu(g,u):
    g=np.minimum(g,7); return g/(1+np.exp(-1.702*g))*(np.clip(u,-7,7)+1)

def reference(w,tokens):
    c=CFG; state=[([],[],[]) for _ in range(c["layers"])]; outputs=[]; selections=[]
    for pos,token in enumerate(tokens):
        x=w["token_embd.weight"][token].copy()
        for l,(keys,values,indices) in enumerate(state):
            p=f"blk.{l}."; h=rms(x,w[p+"attn_norm.weight"])
            q=rotary(rms((w[p+"attn_q.weight"]@h).reshape(c["heads"],c["dim"]),w[p+"attn_q_norm.weight"]),pos)
            k=rotary(rms((w[p+"attn_k.weight"]@h).reshape(c["kv_heads"],c["dim"]),w[p+"attn_k_norm.weight"]),pos)
            keys.append(k); values.append((w[p+"attn_v.weight"]@h).reshape(c["kv_heads"],c["dim"]))
            # M3.1's indexer also runs when this layer's FFN is dense.
            iq=rotary(rms((w[p+"indexer.q_proj.weight"]@h).reshape(c["index_heads"],c["index_dim"]),w[p+"indexer.q_norm.weight"]),pos)
            ik=rotary(rms(w[p+"indexer.k_proj.weight"]@h,w[p+"indexer.k_norm.weight"]),pos)
            indices.append(ik); scores=iq@np.asarray(indices).T
            nb=pos//c["block"]+1
            bs=np.stack([scores[:,b*c["block"]:min((b+1)*c["block"],pos+1)].max(axis=-1) for b in range(nb)],axis=-1)
            if c["local"]: bs[:,-c["local"]:]+=1e30
            chosen=np.argsort(-bs,axis=-1,kind="stable")[:,:min(nb,c["top_blocks"])]
            mask=np.stack([np.isin(np.arange(pos+1)//c["block"],chosen[h]) for h in range(c["kv_heads"])])
            selections.append(chosen.tolist())
            out=[]
            for head in range(c["heads"]):
                kh=head//(c["heads"]//c["kv_heads"])
                score=np.asarray(keys)[:,kh]@q[head]/np.sqrt(np.float32(c["dim"]))
                score=np.where(mask[kh],score,-np.inf); prob=np.exp(score-score.max()); prob/=prob.sum()
                out.append(prob@np.asarray(values)[:,kh])
            x=x+w[p+"attn_output.weight"]@np.concatenate(out)
            h=rms(x,w[p+"ffn_norm.weight"])
            def ffn(suffix,e=None):
                gate=w[p+"ffn_gate"+suffix+".weight"]; up=w[p+"ffn_up"+suffix+".weight"]; down=w[p+"ffn_down"+suffix+".weight"]
                if e is not None: gate,up,down=gate[e],up[e],down[e]
                return down@swiglu(gate@h,up@h)
            if l<c["dense_layers"]: x=x+ffn("")
            else:
                weights=1/(1+np.exp(-(w[p+"ffn_gate_inp.weight"]@h)))
                ids=np.argsort(-(weights+w[p+"exp_probs_b.bias"]),kind="stable")[:c["active"]]
                scale=c["expert_scale"]*weights[ids]/weights[ids].sum()
                x=x+sum(s*ffn("_exps",int(e)) for s,e in zip(scale,ids))+ffn("_shexp")
        outputs.append(w["output.weight"]@rms(x,w["output_norm.weight"]))
    return np.asarray(outputs,dtype=np.float32), selections

def main():
    p=argparse.ArgumentParser(); p.add_argument("--engine"); p.add_argument("--gpu",default="0"); p.add_argument("--write-fixtures",type=pathlib.Path)
    args=p.parse_args()
    with (nullcontext(str(args.write_fixtures)) if args.write_fixtures else tempfile.TemporaryDirectory()) as temporary:
        root=args.write_fixtures or pathlib.Path(temporary); root.mkdir(parents=True,exist_ok=True)
        for mode in ("f32","bf16","q8","nvfp4"):
            model=root/f"tiny-{mode}.gguf"; w=fixture(model,mode); expected,selections=reference(w,TOKENS)
            expected.tofile(root/f"tiny-{mode}.expected.f32")
            if args.engine:
                output=root/f"tiny-{mode}.actual.f32"
                cmd=[args.engine,"--model",str(model),"--tokens",",".join(map(str,TOKENS)),"--context","32","--cache-mib","64","--gpu",args.gpu,"--dump-all-logits",str(output)]
                subprocess.run(cmd,check=True)
                got=np.fromfile(output,dtype=np.float32).reshape(expected.shape)
                np.testing.assert_allclose(got,expected,atol=5e-5,rtol=5e-4)
                print(json.dumps(dict(mode=mode,positions=len(TOKENS),max_abs_error=float(np.max(np.abs(got-expected))),gpu=args.gpu)),flush=True)
        if not args.engine: print(f"Fixtures and independent expected logits written to {root}")
if __name__=="__main__": main()
