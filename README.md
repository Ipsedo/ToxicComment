# ToxicComment

Berrien Samuel, Biard David, Nour Nouredine, Vu-Thanh Trung

## Description

Projet de l'UE Option 7 Apprentissage sur données structurées

Challenge Kaggle [Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

__Structure du projet__ :
```
ToxicComment
    |-- data
    |    |-- read_data.py # Lecture csv
    |    |-- transform_data.py # Différente transformation des données
    |
    |-- model
    |    |-- conv_mode.py # Implémentation des modèle à convolution
    |    |-- recurrent_model.py # i=Implémentation du LSTM
    |
    |-- saved # Different model serializé avec pickle
    |    |-- ...
    |
    |-- utils
    |    |-- cuda.py # Fonction utiles pour CUDA
    |
    |-- live_prediction.ipynb # Pour la prédiction en direct de phrase de l'utilisateur
    |-- make_submission.ipynb # Création d'un fichier de soumission selon un modèle sauvegardé
    |-- plots.ipynb # Création des figures
    |-- toxic_comment.ipynb # Notebook principal d'entrainement
```

Les différents csv sont à placer dans un répertoire `res` situé à la racine soit `/path/to/ToxicComment/res/`