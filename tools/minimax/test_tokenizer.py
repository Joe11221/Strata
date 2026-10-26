"""Compare real GGUF token ids with the official Hugging Face tokenizer.json.

Requires regex + tokenizers. No model weights or GPU are needed.
python tools/minimax/test_tokenizer.py METADATA_SHARD OFFICIAL_TOKENIZER_JSON
"""
import json
import pathlib
import random
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from strata_tokenizer import Tokenizer

def main():
    from tokenizers import Tokenizer as OfficialTokenizer
    native=Tokenizer.from_gguf(pathlib.Path(sys.argv[1]))
    reference=OfficialTokenizer.from_file(sys.argv[2])
    cases=["Hello, world!","XMLHttpRequest ABCDef HTTPServer","1234567890",
           "caf\u00e9 e\u0301 \u4e2d\u6587 \u65e5\u672c\u8a9e \u0627\u0644\u0639\u0631\u0628\u064a\u0629",
           "\U0001f98a\U0001f642\n\t  x  ","I'M I'll can't doesn't","a_b::foo/bar/\r\n",
           "<mm:think>42</mm:think>","]~b]user\nHello[e~["]
    rng=random.Random(20261026)
    alphabet="abcXYZ0123456789 _.,/\n\t!?\u00e9\u4e2d\U0001f642"
    cases += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1,180))) for _ in range(500)]
    for text in cases:
        got=native.encode(text,parse_special=True)
        want=reference.encode(text,add_special_tokens=False).ids
        if got!=want: raise AssertionError(f"token mismatch: {text!r}\n{got}\n{want}")
    print(json.dumps({"cases":len(cases),"mismatches":0,"pre":native.pre}))
if __name__=="__main__": main()
