# Charger les modules nécessaires
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords # module qui contient une liste de mots courants

MOTS_COURANTS = "mots_courants.txt"
PATH_HUMAN_TEXTS = "/humain/"
PATH_AI_TEXTS = "/ia/"

# Fonction qui lit plusieurs fichiers texte dans PATH et les convertis une chaine de caractères
def read_texts(a):
    chaine = []
    # chemin courant
    path = os.getcwd()
    if a==1 :
        path = path+PATH_HUMAN_TEXTS
    else :
        path = path+PATH_AI_TEXTS
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            chaine.append(read_one_text(path + filename))
    return chaine

# Fonction qui lit un fichier texte et le convertis en une chaine de caractères
def read_one_text(filename):
    print("lecture du fichier " + filename + "...")
    texts = []
    with open(filename, "r", encoding='utf-8') as file:
        for line in file:
            texts.append(line)
    file.close()
    return "".join(texts)

# Fonction qui écrit dans un fichier texte les mots courants français et créer le fichier s'il n'existe pas, sans utiliser with open
def write_common_words(filename):
    common_words = set(stopwords.words('French'))
    file = open(filename, "w")
    for word in common_words:
        file.write(word + "\n")
    file.close()

# Fonction qui lit un fichier texte et le stocke dans mots_courants
def read_common_words(filename):
    common_words = []
    with open(filename, "r") as file:
        for line in file:
            common_words.append(line.strip()) # .strip() enlève les espaces avant et après la chaine de caractères
    file.close()
    return common_words

# Extraire les caractéristiques des textes
# Fonction qui donne le nombre de mots dans un texte
def count_words(text):
    return len(text.split())
    
# Fonction renvoie la longueur moyenne des phrases
def average_sentence_length(text):
    # Séparer le texte en phrases qui sont séparées par un point ou un point d'exclamation ou un point d'interrogation ou un point-virgule ou un deux-points
    sentences = text.split(".") + text.split("!") + text.split("?") + text.split(";") + text.split(":")
    return sum([len(sentence.split()) for sentence in sentences]) / len(sentences)

# Fonction qui donne le nombre de mots uniques utilisés dans un texte.
def count_unique_words(text):
    return len(set(text.split()))

# Fonction qui donne la répétition des mots
def count_repeated_words(text):
    return len(text.split()) - len(set(text.split()))

# Fonction qui donne le ratio de mots rares utilisés dans un texte
# une valeur de 0.5 signifie que la moitié des mots utilisés sont rares
# une valeur de 0.1 signifie que 10% des mots utilisés sont rares
def rare_word_use(text):
    common_words = read_common_words(MOTS_COURANTS)
    words = text.split() # séparer le texte en mots
    num_words = len(words) # nombre de mots
    num_rare_words = len([m for m in words if m not in common_words])# calcule le nombre de mots rares dans le texte (mots qui ne sont pas dans la liste de mots courants)
    if num_words == 0:
        return 0
    return num_rare_words / num_words

# PROGRAMME PRINCIPAL

# Charger les textes d'entraînement
texts_from_ai = read_texts(0)
texts_from_human = read_texts(1)

# Ecrire dans un fichier les mots courants
write_common_words(MOTS_COURANTS)

# Définir les labels correspondants pour chaque texte (0 pour IA, 1 pour humain)
# labels doit contenir autant d'éléments que de textes d'entraînement (len(texts_from_ai) + len(texts_from_human))
labels = [0] * len(texts_from_ai) + [1] * len(texts_from_human)

# Extraire les caractéristiques des textes (nombre de mots, longueur moyenne des phrases, et nombre de mots uniques, répétition des mots)
num_words = []
avg_sentence_length = []
num_unique_words = []
num_repeat_words = []
num_rare_word_use = []
print("Entraînement du modèle...")
for texts in [texts_from_ai, texts_from_human]:
    num_words.extend([count_words(text) for text in texts])#.extend permet d'ajouter les éléments d'une liste à une autre
    avg_sentence_length.extend([average_sentence_length(text) for text in texts])
    num_unique_words.extend([count_unique_words(text) for text in texts])
    num_repeat_words.extend([count_repeated_words(text) for text in texts])
    num_rare_word_use.extend([rare_word_use(text) for text in texts])

# Créer un modèle de régression logistique
model = LogisticRegression()

# Entraîner le modèle en utilisant les textes et les labels
model.fit(np.column_stack([num_words, avg_sentence_length, num_unique_words, num_repeat_words, num_rare_word_use]), labels)
# .fit permet d'entraîner le modèle en utilisant les données d'entraînement et les labels correspondants (les labels sont les bonnes réponses)

# Prédire si le contenu entier d'un fichier texte provient d'une IA ou d'un humain
new_text = read_one_text("new_text.txt")
num_words = count_words(new_text)
avg_sentence_length = average_sentence_length(new_text)
num_unique_words = count_unique_words(new_text)
num_repeat_words = count_repeated_words(new_text)
num_rare_word_use = rare_word_use(new_text)
# Il faut que les données d'entraînement et les données à prédire aient le même nombre de dimensions
# Ici, les données d'entraînement sont des tableaux à deux dimensions et les données à prédire sont des tableaux à une dimension
# Pour transformer un tableau à une caractéristique en tableau à trois caractéristiques, on utilise la fonction column_stack
print("Prédiction: ")
X = np.column_stack([num_words, avg_sentence_length, num_unique_words, num_repeat_words, num_rare_word_use])
prediction = model.predict(X)
# donne la prédiction avec sa probabilité
probabilities = model.predict_proba(X)
proba_arrondi_IA = round(probabilities[0][0], 4)
proba_arrondi_humain = round(probabilities[0][1], 4)

if prediction == 0:
    print("Ce texte a été généré par une IA avec une probabilité de", proba_arrondi_IA * 100, "%")
else:
    print("Ce texte a été écrit par un humain avec une probabilité de", proba_arrondi_humain * 100, "%")



