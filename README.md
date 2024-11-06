# Classification-d-Images-Simple
Description
Ce projet, créé par Mouhammad et Deo, met en œuvre une pipeline de machine learning pour la reconnaissance faciale et la classification d'images. Le pipeline inclut la détection de visages, l'extraction de caractéristiques faciales, la normalisation des données, et l'entraînement d'un modèle de réseau de neurones. Le projet utilise des bibliothèques populaires telles que OpenCV, Scikit-learn, Keras, et FaceNet de Pytorch.

Table des Matières:

Installation
Utilisation
Structure du Projet
Détails du Pipeline
Résultats
Contributions
License
Installation
Prérequis
Python 3.x
Jupyter Notebook
Bibliothèques Python nécessaires
Installez les bibliothèques nécessaires à l'aide de pip :
pip install numpy pandas scikit-learn tensorflow keras facenet-pytorch mtcnn opencv-python
Utilisation
Clonez ce dépôt sur votre machine locale :
git clone https://github.com/votre-utilisateur/classification-images-simple.git
Accédez au répertoire du projet :

cd classification-images-simple
jupyter notebook
Ouvrez le fichier classification_images_simple.ipynb et exécutez toutes les cellules pour reproduire le pipeline et les résultats.

Structure du Projet
bash
Copier le code
classification-images-simple/
├── data/
│   └── images
├── classification_images_simple.ipynb
├── README.md
└── requirements.txt
data/: Répertoire pour les fichiers de données.
classification_images_simple.ipynb: Notebook Jupyter contenant le code du projet.
README.md: Ce fichier.
requirements.txt: Liste des dépendances nécessaires pour exécuter le projet.
Détails du Pipeline
Chargement des images et détection des visages :
Les images sont chargées depuis le répertoire spécifié et les visages sont détectés à l'aide de MTCNN.

Prétraitement des images pour FaceNet :
Les images sont redimensionnées, normalisées et transposées pour correspondre au format attendu par FaceNet.

Extraction des caractéristiques faciales avec FaceNet :
Les caractéristiques des visages sont extraites à l'aide du modèle InceptionResnetV1 préentraîné de FaceNet.

Prétraitement des données pour l'entraînement du modèle :
Les étiquettes des images sont encodées et converties en format one-hot.

Division des données en ensembles d'entraînement et de test :
Les données sont divisées en ensembles d'entraînement et de test.

Construction et entraînement du modèle de classification :
Un modèle de réseau de neurones est construit et entraîné sur les caractéristiques extraites des images.

Évaluation et enregistrement du modèle :
Le modèle est évalué sur les données de test et enregistré.

Résultats
Le modèle de classification est évalué sur les données de test et les métriques de performance sont affichées. Le modèle entraîné est également sauvegardé pour une utilisation future.

Contributions
Les contributions sont les bienvenues ! Pour contribuer :

Fork ce dépôt.
Créez une branche pour votre fonctionnalité (git checkout -b fonctionnalite-xy).
Commitez vos modifications (git commit -m 'Ajouter fonctionnalite-xy').
Poussez vers la branche (git push origin fonctionnalite-xy).
Ouvrez une Pull Request.

