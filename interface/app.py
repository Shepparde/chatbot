import random
import re
from flask import Flask, render_template, request
from chat import get_recommendations_tfidf,tfidf_mat
import pickle
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from preprocessing import applyLemming,nlp,Remove_Punct

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
#function for the bot response
def get_bot_response():
    userText = request.args.get('msg')
    userText = applyLemming(nlp(Remove_Punct(userText)))
    userText=" ".join(userText)
    print(userText)
    salutationsAnswers = re.compile('bonjour|bjr|hello|hi')
    if re.search(salutationsAnswers, userText.lower()) and len(userText) < 10:
        return str("Bonjour, <br> pouvez-vous me décrire en détail vos envies d'activités")
    else:
        return str(get_recommendations_tfidf(userText.lower(),tfidf_mat))
    
if __name__ == "__main__":
    app.run()
