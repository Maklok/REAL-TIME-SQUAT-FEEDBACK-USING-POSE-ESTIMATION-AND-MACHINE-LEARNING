1.Open terminal in the folder
2.Create virtual environment:
  python -m venv venv
3.Activate it:
  Windows:
  venv\Scripts\activate
4.Install dependencies:
  pip install -r requirements.txt
6.(Optional) Add Gemini key:
  set GOOGLE_API_KEY=AIzaSyDCewBkMeDxUByP4AySCXyWKEcGQ1tqd9w
7.Run the app:
  uvicorn app:app
8.Open browser:
  http://127.0.0.1:8000/docs