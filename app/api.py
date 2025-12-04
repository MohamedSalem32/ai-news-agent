# fast-reading-agent/app/api.py
from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
from .tools import FastReaderCore

app = FastAPI(title="Fast Reading (Digest & Focus) API")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CORE = FastReaderCore(DATA_DIR, use_st=False)

class FocusReq(BaseModel):
    query: str
    top_n: int = 3

class DigestReq(BaseModel):
    query: str
    n: int = 6

@app.post("/focus")
def focus(req: FocusReq):
    res = CORE.focus(req.query, req.top_n)
    return {"results": res}

@app.post("/digest")
def digest(req: DigestReq):
    res = CORE.digest(req.query, req.n)
    return {"results": res}
