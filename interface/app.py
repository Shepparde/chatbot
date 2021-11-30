import random
import pickle
import json
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    
    return str("bot")


if __name__ == "__main__":
    app.run()
