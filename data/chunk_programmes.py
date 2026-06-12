# data/chunk_programmes.py
import json
import tiktoken
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent
TEXT_DIR = DATA_DIR / "wahlprogramme" / "text"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Use cl100k_base tokenizer (same as GPT-4 / similar to Llama tokenizer length)
enc = tiktoken.get_encoding("cl100k_base")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=256,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""]
)

for txt_path in sorted(TEXT_DIR.glob("*.txt")):
    party = txt_path.stem  # e.g. "cdu_csu_2025"
    text = txt_path.read_text(encoding="utf-8")
    chunks = splitter.split_text(text)

    out_path = CHUNKS_DIR / f"{party}_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "party": party,
                "chunk_id": i,
                "text": chunk,
                "token_count": len(enc.encode(chunk))
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"{party}: {len(chunks)} chunks (avg {sum(len(enc.encode(c)) for c in chunks)//len(chunks)} tokens)")