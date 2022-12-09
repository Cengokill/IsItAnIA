import os

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model

MOTS_COURANTS = "mots_courants.txt"
PATH_HUMAN_TEXTS = "/humain/"
PATH_AI_TEXTS = "/ia/"
NEW_TEXT = "new_text.txt"
CURRENTPATH = os.getcwd()

# Fonction qui lit plusieurs fichiers texte dans PATH et les convertis en une chaine de caractères
def read_texts(a):
    chaine = []
    # chemin courant
    chemin = CURRENTPATH
    if a==1 :
        chemin = chemin+PATH_HUMAN_TEXTS
    else :
        chemin = chemin+PATH_AI_TEXTS
    for filename in os.listdir(chemin):
        if filename.endswith(".txt"):
            chaine.append(read_one_text(chemin + filename))
    return chaine

# Fonction qui lit un fichier texte et le convertis en une chaine de caractères
def read_one_text(filename):
    texts = []
    with open(filename, "r", encoding='utf-8') as file:
        for line in file:
            texts.append(line)
    file.close()
    return "".join(texts)

def text_to_vectors(text):
  # Convertissons les caractères en vecteurs de 0 et 1
  # 1 indique que le caractère est présent dans le texte
  # 0 indique qu'il n'y est pas
  vectors = []
  for char in text:
    if char in text:
      vectors.append(1)
    else:
      vectors.append(0)
  # Retournons le vecteur de caractères
  return np.array(vectors)

# Charger les données d'entraînement
human_texts = read_texts(1)  # lire les textes écrits par des humains
ai_texts = read_texts(0)  # lire les textes générés par des IA

# Convertir les textes en vecteurs de caractères
human_text_vectors = text_to_vectors(human_texts)
ai_text_vectors = text_to_vectors(ai_texts)

# Définissons les données d'entraînement pour le modèle
x_train = np.array([ai_text_vectors,human_text_vectors])
y_train = np.array([1, 0,]) # 1 pour IA, 0 pour humain

# Définissons la structure du modèle.
# Nous utilisons ici un réseau de neurones à une couche cachée
inputs = Input(shape=(x_train.shape[1],))
hidden = Dense(16, activation="relu")(inputs)
output = Dense(1, activation="sigmoid")(hidden)

# Compilons le modèle
model = Model(inputs=inputs, outputs=output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Entraînons le modèle
BATCH_SIZE = 32
EPOCHS = 20

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Utilisons le modèle pour prédire si un texte a été généré par une IA ou non
def is_text_generated_by_ai(text):
  # Convertissons le texte en vecteur de caractères
  x = text_to_vectors(text)
  # Utilisons le modèle pour prédire si le texte a été généré par une IA ou non
  y = model.predict(x)
  # Retournons True si le texte a été généré par une IA, False sinon
  return y >= 0.5
