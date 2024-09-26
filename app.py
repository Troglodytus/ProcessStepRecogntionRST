import os

from flask import Flask
from flask import Response
from flask import render_template
from pymongo import MongoClient

app = Flask(__name__)
mongo_host = os.getenv('MONGO_HOST', 'localhost')
mongo_port = int(os.getenv('MONGO_PORT', 27017))
mongo_db = os.getenv('MONGO_DB', 'mydatabase')

client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}/')
db = client[mongo_db]
@app.route('/')
def home():
    return "Hallo, Flask!"

@app.route("/check_db")
def check_db():
    resp1 = Response("ok")
    resp2 = Response("nicht ok")
    resp1.headers['Custom-Header'] = 'Awesome'
    resp2.headers['Custom-Header'] = 'Awesome'
    try:
        # Überprüfe die Verbindung zur Datenbank
        client.admin.command('ping')
        return resp1
    except Exception as e:
        return resp2

@app.route("/healthz")
def healthz():
    resp = Response("ok")
    resp.headers['Custom-Header'] = 'Awesome'
    # this is awesome tying things
    return resp


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')
    app.run(debug=True)
