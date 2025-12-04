'''
from __future__ import annotations
import re
from typing import Tuple

_SUMMARIZER = None
_TOKENIZER = None

def _get_model() -> Tuple[object, object]:
    global _SUMMARIZER, _TOKENIZER
    if _SUMMARIZER is None or _TOKENIZER is None:
        from transformers import pipeline, AutoTokenizer
        model_id = "sshleifer/distilbart-cnn-12-6"
        # device=-1 => CPU; set to 0 if you have CUDA
        _SUMMARIZER = pipeline("summarization", model=model_id, device=-1)
        _TOKENIZER = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    return _SUMMARIZER, _TOKENIZER

def _truncate_tokens(text: str, max_tokens: int = 1000) -> str:
    """
    Truncate input by tokens to stay under the model's 1024-token limit.
    """
    _, tok = _get_model()
    enc = tok(text, truncation=True, max_length=max_tokens, return_attention_mask=False, return_token_type_ids=False)
    ids = enc["input_ids"]
    # decode back to text (skip special tokens)
    return tok.decode(ids, skip_special_tokens=True)

def _postprocess_summary(s: str, word_cap: int = 40) -> str:
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    words = s.split()
    if len(words) > word_cap:
        s = " ".join(words[:word_cap]).rstrip(",.;:") + "…"
    return s

def summarize_40w(text: str) -> str:
    summarizer, _ = _get_model()

    # Hard cap the input by tokens (DistilBART max is 1024; keep margin)
    chunk = _truncate_tokens(text, max_tokens=1000)

    # Dynamic output lengths for short/long inputs
    wc = len(chunk.split())
    max_len = max(10, min(30, wc - 3))   # clamp to [10..30]
    min_len = max(8, min(12, max_len - 8))

    out = summarizer(
        chunk,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        early_stopping=True,
        truncation=True,   # be explicit even though we pre-truncated
    )[0]["summary_text"]

    return _postprocess_summary(out, word_cap=40)
'''

from __future__ import annotations
from typing import Optional
import re

# Global variable to hold the initialized HuggingFace pipeline
_SUMMARY_PIPE = None

# --- Core Helper Functions ---

_WS = re.compile(r"\s+")

def _clip_40w(text: str) -> str:
    """
    Clips a string to a maximum of 40 words, appending an ellipsis if clipped.
    """
    # Use re.findall to correctly handle various whitespace and punctuation splits
    words = re.findall(r"\S+", text or "")
    
    if len(words) <= 40: 
        return " ".join(words)
    
    # Join the first 40 words and append ellipsis
    return " ".join(words[:40]).rstrip(",.;:") + "…"

def _lead_extract_40w(text: str) -> str:
    """
    Fast extractive fallback: first ~40 words from the body.
    """
    return _clip_40w((text or "").strip())

def _load_en_pipeline():
    """
    Loads and caches the DistilBART summarization pipeline.
    """
    global _SUMMARY_PIPE
    if _SUMMARY_PIPE is not None:
        return _SUMMARY_PIPE
    
    try:
        # We only import transformers here, so the module loads even if it's missing
        from transformers import pipeline
        # device=-1 tells HuggingFace to use CPU (default)
        _SUMMARY_PIPE = pipeline(
            "summarization", 
            model="sshleifer/distilbart-cnn-12-6", 
            device=-1
        )
    except Exception:
        # If transformers/torch is missing, set to None so we use the fallback
        _SUMMARY_PIPE = None
        # You could add logging here to alert that the model failed to load
        
    return _SUMMARY_PIPE

# --- Public Export ---

def summarize_40w(text: str, lang: str = "en") -> str:
    """
    Guaranteed to return a summary of at most 40 words.
    Uses DistilBART for English, or a fast extractive fallback otherwise.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # 1. Non-English and No-Model Fallback Path
    if (lang or "en").lower() != "en":
        # Quick fallback for French, German, etc.
        return _lead_extract_40w(text)
        
    pipe = _load_en_pipeline()
    if not pipe:
        # Fallback if model could not be loaded (e.g., dependencies missing)
        return _lead_extract_40w(text)

    # 2. English (DistilBART) Path
    
    # DistilBART has a token limit (around 1024), so clip the input text 
    # slightly below that threshold to be safe.
    if len(text) > 3000:
        text = text[:3000]

    try:
        # Use a short max_length to encourage a concise summary suitable for clipping
        out = pipe(
            text, 
            max_length=120, 
            min_length=60, 
            do_sample=False,
            num_beams=4,
            length_penalty=1.0,
            truncation=True
        )[0]["summary_text"]
        
        # Clip the generated summary to the 40-word hard limit
        return _clip_40w(out)
    except Exception:
        # Fallback if summarization fails (e.g., context-specific error)
        return _lead_extract_40w(text)