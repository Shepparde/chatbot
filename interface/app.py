import random
import re


from flask import Flask, render_template, request

import chat

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    
    salutationsAnswers = re.compile('bonjour|bjr|hello|hi')
    if re.search(salutationsAnswers, userText.lower()) and len(userText) < 10:
        return str("Bonjour, <br> pouvez-vous me décrire en détail vos envies d'activités")
    else:
        return str(chat.get_recommendations_tfidf(userText.lower(),chat.tfidf_mat))#str("Je n'ai pas bien compris votre demande, <br>pouvez-vous s'il vous plait me décrire en détail vos envies d'activités?")
    
if __name__ == "__main__":
    app.run()
