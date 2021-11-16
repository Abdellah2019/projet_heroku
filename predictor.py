import sys
import joblib 

#Recupérer le temps de réponse passé en paramètre
responseTime=float(sys.argv[1])
#Charger le modèle depuis le fichier
model=joblib.load("./model.joblib")
#Predire le montant moyen d'achat 
purchasePredict=model.predict([[responseTime]])
print("Prediction={:.4f}".format(purchasePredict[0]))