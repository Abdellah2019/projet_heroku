import joblib 
from flask import Flask
from flask import request

app=Flask(__name__)
@app.route('/<perfRespTime>')
def index(perfRespTime):
	#Charer le modèle depuis le fichier
	model=joblib.load("./model.joblib")
	#Prédire le montant d'achat à l'aide du modèle 
	purchasePredict=model.predict([[float(perfRespTime)]])
	#Afficher le resultat dans le navigateur 
	return ("Prediction={:.4f}".format(purchasePredict[0]))
app.run(host='localhost',port=5000)