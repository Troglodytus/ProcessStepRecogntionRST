from flask import Flask
from flask import Response
from flask import render_template
from pymongo import MongoClient

app = Flask(__name__)
client = MongoClient("mongodb://localhost:27017/")
db = client.test_database
@app.route('/')
def home():
    return "Hallo, Flask!"

@app.route("/check_db")
def check_db():
    try:
        # Überprüfe die Verbindung zur Datenbank
        client.admin.command('ping')
        return "Verbindung besteht!"
    except Exception as e:
        return f"Verbindung fehlgeschlagen: {e}"

@app.route("/healthz")
def healthz():
    resp = Response("ok")
    resp.headers['Custom-Header'] = 'Awesome'
    # this is awesome tying things
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')
