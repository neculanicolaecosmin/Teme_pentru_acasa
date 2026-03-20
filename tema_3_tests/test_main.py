import requests
import httpx
import sys
import pytest
from fastapi.testclient import TestClient

from app.main import app, assistant_instance

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

client = TestClient(app)

def test_root_endpoint_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Salut, RAG Assistant ruleaza!"}


def test_chat_endpoint_relevant_question(monkeypatch):
    # Simulam răspunsul LLM pentru a nu depinde de rețea/API pe test local
    monkeypatch.setattr(assistant_instance, "assistant_response", lambda msg: "Răspuns generat pentru yoga")

    payload = {"message": "Care sunt beneficiile Practicii Asana pentru flexibilitate?"}
    response = client.post("/chat/", json=payload)

    assert response.status_code == 200
    assert "Răspuns generat pentru yoga" in response.json().get("response", "")


def test_chat_endpoint_negative_irrelevant_query():
    payload = {"message": "Ce temperatură e la munte?"}
    response = client.post("/chat/", json=payload)

    assert response.status_code == 200
    assert "nu pare a fi despre yoga" in response.json().get("response", "").lower() 