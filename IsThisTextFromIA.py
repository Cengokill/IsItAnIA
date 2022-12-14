# Charger les modules nécessaires
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords # module qui contient une liste de mots courants

MOTS_COURANTS = "mots_courants.txt"
PATH_HUMAN_TEXTS = "/humain/"
PATH_AI_TEXTS = "/ia/"
NEW_TEXT = "new_text.txt"
CURRENTPATH = os.getcwd()

# Fonction qui recherche une chaine dans un texte en utilisant l'algorithme de Karp-Rabin
def recherche_KarpRabin(text, pattern):
    pattern_hash = hash(pattern)  # calculer le hachage de la chaîne à rechercher
    for i in range(len(text) - len(pattern) + 1):
        # calculer le hachage de la sous-chaîne de longueur égale à celle de la chaîne à rechercher
        sub_string_hash = hash(text[i:i + len(pattern)])
        if pattern_hash == sub_string_hash:  # si les hachages sont égaux, vérifier si les chaînes sont égales
            if text[i:i + len(pattern)] == pattern:
                return True  # retourner True si les chaînes sont égales
    return False  # retourner False si la chaîne n'a pas été trouvée

# Recherche un texte dans plusieurs fichiers
def recherche_chaine_dossier(text, dossier):
    for filename in os.listdir(dossier):
        if filename.endswith(".txt"):
            if recherche_KarpRabin(read_one_text(dossier + filename), text):
                return True
    return False
def save_text(new_text, path):
    print("Enregistrement du texte dans le chemin " + path)
    if len(new_text) < 3000:
        # On ouvre le fichier train.txt qui contient le numéro le plus élevé
        num = 0
        for filename in os.listdir(path):
            if filename.startswith("train"):
                num = int(filename[5])
        with open(path + "train" + str(num) + ".txt", "a",
                  encoding="utf-8") as f:  # l'ouverture en "a" permet d'ajouter du texte à la fin du fichier
            # on ajoute retour à la ligne et le texte
            f.write("\n" + new_text)
        f.close()
    else:
        # On crée un nouveau fichier train.txt
        num = 0
        for filename in os.listdir(path):
            if filename.startswith("train"):
                num = int(filename[5])
        with open(path + "train" + str(num + 1) + ".txt", "w", encoding="utf-8") as f:
            f.write(new_text)
        f.close()
    print("Merci! Le fichier a été enregistré.")
    print("N'oubliez pas de PUSH sur GitHub pour que votre texte soit ajouté à la base de données.")

# Fonction qui lit plusieurs fichiers texte dans PATH et les convertis une chaine de caractères
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

# Fonction qui donne le nombre de mots en majuscule
def count_uppercase_words(text):
    return len([m for m in text.split() if m.isupper()])

# Fonction qui renvoie un tableau de la fréquence des caractères : , ; ! ? - dans le texte
def frequency_character(text):
    longueur = len(text)
    tableau = [0, 0, 0, 0, 0, 0]

    tableau.append(text.count(":") / longueur)
    tableau.append(text.count(",") / longueur)
    tableau.append(text.count(";") / longueur)
    tableau.append(text.count("!") / longueur)
    tableau.append(text.count("?") / longueur)
    tableau.append(text.count("-") / longueur)
    return tableau

# ENTRAINEMENT
# Fonction qui entraine le modèle
def train_model():
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
    num_uppercase_words = []
    frequency_two_points = []
    frequency_comma = []
    frequency_semicolon = []
    frequency_exclamation = []
    frequency_question = []
    frequency_dash = []
    print("Entraînement du modèle...")
    for texts in [texts_from_ai, texts_from_human]:
        num_words.extend(
            [count_words(text) for text in texts])  # .extend permet d'ajouter les éléments d'une liste à une autre
        avg_sentence_length.extend([average_sentence_length(text) for text in texts])
        num_unique_words.extend([count_unique_words(text) for text in texts])
        num_repeat_words.extend([count_repeated_words(text) for text in texts])
        num_rare_word_use.extend([rare_word_use(text) for text in texts])
        num_uppercase_words.extend([count_uppercase_words(text) for text in texts])
        frequency_two_points.extend([frequency_character(text)[0] for text in texts])
        frequency_comma.extend([frequency_character(text)[1] for text in texts])
        frequency_semicolon.extend([frequency_character(text)[2] for text in texts])
        frequency_exclamation.extend([frequency_character(text)[3] for text in texts])
        frequency_question.extend([frequency_character(text)[4] for text in texts])
        frequency_dash.extend([frequency_character(text)[5] for text in texts])

    # Mettre à l'échelle les données
    scaler = StandardScaler()

    # Adapter l'échelle aux données
    scaled_data = scaler.fit_transform(scaler.fit_transform(np.column_stack([num_words, avg_sentence_length, num_unique_words, num_repeat_words, num_rare_word_use, num_uppercase_words,
                               frequency_two_points, frequency_comma, frequency_semicolon, frequency_exclamation, frequency_question, frequency_dash])))

    # Créer un modèle de régression logistique
    model = LogisticRegression(max_iter=200)

    # Entraîner le modèle sur les données mises à l'échelle
    model.fit(scaled_data, labels)

    print("Entraînement terminé.")
    # .fit permet d'entraîner le modèle en utilisant les données d'entraînement et les labels correspondants (les labels sont les bonnes réponses)
    return model

# PROGRAMME PRINCIPAL
if __name__ == "__main__":
    # Si le fichier new_text.txt est vide ou ne contient pas de texte, on quitte le programme
    if not os.path.exists(NEW_TEXT) or os.stat(NEW_TEXT).st_size == 0:
        print("Veuillez écrire un texte dans le fichier new_text.txt")
        sys.exit(0)

    # Entrainer le modèle
    model = train_model()

    # Prédire si le contenu entier d'un fichier texte provient d'une IA ou d'un humain
    new_text = read_one_text("new_text.txt")
    num_words = count_words(new_text)
    avg_sentence_length = average_sentence_length(new_text)
    num_unique_words = count_unique_words(new_text)
    num_repeat_words = count_repeated_words(new_text)
    num_rare_word_use = rare_word_use(new_text)
    num_uppercase_words = count_uppercase_words(new_text)
    freq = frequency_character(new_text)
    frequency_two_points = freq[0]
    frequency_comma = freq[1]
    frequency_semicolon = freq[2]
    frequency_exclamation = freq[3]
    frequency_question = freq[4]
    frequency_dash = freq[5]
    # Il faut que les données d'entraînement et les données à prédire aient le même nombre de dimensions
    # Ici, les données d'entraînement sont des tableaux à deux dimensions et les données à prédire sont des tableaux à une dimension
    # Pour transformer un tableau à une caractéristique en tableau à trois caractéristiques, on utilise la fonction column_stack
    print("Prédiction: ")
    X = np.column_stack([num_words, avg_sentence_length, num_unique_words, num_repeat_words, num_rare_word_use, num_uppercase_words, frequency_two_points, frequency_comma, frequency_semicolon, frequency_exclamation, frequency_question, frequency_dash])
    prediction = model.predict(X)
    # donne la prédiction avec sa probabilité
    probabilities = model.predict_proba(X)
    proba_arrondi_IA = round(probabilities[0][0]*100, 2)
    proba_arrondi_humain = round(probabilities[0][1]*100, 2)

    textFromIA = 0
    if prediction == 0:
        print("Ce texte a été généré par une IA avec une probabilité de", proba_arrondi_IA, "%")
        # Il y a plusieurs raisons pour lesquelles votre programme pourrait toujours détecter que les textes sont écrits par une IA.
        # Tout d'abord, il est possible que vous n'ayez pas suffisamment de données d'entraînement pour entraîner le modèle de manière précise.
        # Dans ce cas, vous pourriez obtenir de meilleurs résultats en ajoutant plus de données d'entraînement à vos dossiers.
        #
        # Il est également possible que votre fonction de prétraitement des données ne soit pas efficace pour nettoyer les textes.
        # Si votre fonction ne supprime pas suffisamment de mots inutiles, le modèle peut avoir du mal à différencier les textes écrits par des humains et des intelligences artificielles.
        # Vous pourriez essayer de modifier la fonction pour qu'elle supprime davantage de mots inutiles ou utilise des techniques de prétraitement différentes.
        #
        # Enfin, il est possible que le modèle de régression logistique ne soit pas adapté à vos données. Dans ce cas, vous pourriez
        # essayer d'utiliser un autre algorithme de machine learning pour entraîner votre modèle. Vous pouvez par exemple essayer de mettre
        # en place un réseau de neurones pour voir si les résultats s'améliorent.
        textFromIA = 0
    else:
        print("Ce texte a été écrit par un humain avec une probabilité de", proba_arrondi_humain, "%")
        textFromIA = 1

    # Demande à l'utilisateur si c'est correct
    print("Est-ce correct? (O/N)")
    correct = input()
    if correct == "O" or correct == "o":
        print("Merci!")
        # On vide le contenu de new_text.txt
        with open("new_text.txt", "w") as f:
            f.write("")
            f.close()
        sys.exit(0)
    else:
        if textFromIA == 0:
            path = CURRENTPATH + PATH_HUMAN_TEXTS
        else:
            path = CURRENTPATH + PATH_AI_TEXTS
        # On recherche le texte new_text.txt en utilisant la fonction search(text, pattern) dans tous les fichiers textes d'entraînement situés dans path
        # Si le texte est trouvé, on arrête le programme. Sinon, on enregistre le texte dans un nouveau fichier
        if recherche_chaine_dossier(new_text, path):
            print("Erreur : ce texte a déjà été enregistré.")
        else:
            # On crée un fichier qui sera enregistré dans PATH_HUMAN_TEXTS si textFromIA = 1 et dans PATH_AI_TEXTS si textFromIA = 0
            save_text(new_text, path)
    #On vide le contenu de new_text.txt
    with open("new_text.txt", "w") as f:
        f.write("")
        f.close()
# Sais-tu ce qui a causé la destruction de la fusée Ariane 5 en 1996 ?
# En réalité, c'est un problème de logiciel. Un dépassement d'entier a causé la destruction de la fusée.
# Le problème a été causé par un programmeur qui a utilisé un entier de 32 bits au lieu d'un entier de 64 bits.
# Cela a causé un débordement de mémoire et a causé la destruction de la fusée.