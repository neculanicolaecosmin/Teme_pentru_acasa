# TEME_PENTRU_ACASA

Instrucțiuni:

1. Rulați în terminal:
python -m venv venv
. .\venv\Scripts\activate
pip install -r .\requirements.txt

2. Redenumiți fișierul .env.sample în .env și completați variabilele cu valorile dvs.

3. Instalare extra dependențe necesare:
python -m pip install --upgrade pip
pip install tensorflow_hub

4. Testare:
python src/tema_2_services/service.py

4. Pentru a rula testele din test_main rulați:
în primul terminal: uvicorn app.main:app --reload
și în al doilea terminal: pytest

5. Pentru a rula metricile din evaluate rulați:
în primul terminal: uvicorn app.main:app --reload
și în al doilea terminal: python -m tema_3_evaluation.evaluate
