"""Minimal FastAPI starter (own code, MIT)."""
from fastapi import FastAPI

app = FastAPI(title="fastapi-min")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {"hello": "fastapi-min"}
