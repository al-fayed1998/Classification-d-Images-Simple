#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 00:15:15 2023

@author: deo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 23:46:42 2023

@author: deo
"""
import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from facenet_pytorch import InceptionResnetV1
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from mtcnn import MTCNN





from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import os
import cv2
import numpy as np
from sklearn.utils import shuffle
from mtcnn import MTCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from facenet_pytorch import InceptionResnetV1

# Chemin d'accès au dossier contenant les images à prétraiter
image_folder_path = '/home/deo/.config/spyder-py3/vraie images'

# Initialiser le détecteur MTCNN
detector = MTCNN()

# Charger les images, détecter les caractéristiques du visage et prétraiter les données
def load_images_and_preprocess(image_folder_path):
    images = []
    labels = []
    
    for label_name in os.listdir(image_folder_path):
        label_folder_path = os.path.join(image_folder_path, label_name)
        
        if os.path.isdir(label_folder_path):
            for image_file in os.listdir(label_folder_path):
                image_path = os.path.join(label_folder_path, image_file)
                image = cv2.imread(image_path)
                
                if image is not None:
                    faces = detector.detect_faces(image)
                    
                    for face in faces:
                        x, y, w, h = face['box']
                        face_image = image[y:y+h, x:x+w]
                        resized_face = cv2.resize(face_image, (160, 160))
                        
                        images.append(resized_face)
                        labels.append(label_name)
                        
    return images, labels

images, labels = load_images_and_preprocess(image_folder_path)
images, labels = shuffle(images, labels, random_state=42)

# Charger le modèle FaceNet préentraîné
face_model = InceptionResnetV1(pretrained='vggface2').eval()

# Prétraiter les images pour les entrées du modèle FaceNet
images = np.array(images)
images = images.astype('float32') / 255.0
images = np.transpose(images, (0, 3, 1, 2))

# Extraire les caractéristiques avec FaceNet
features = []
for img in images:
    img = np.expand_dims(img, axis=0)
    img_embedding = face_model(torch.from_numpy(img))
    features.append(img_embedding.detach().numpy())

flattened_features = np.array(features).reshape(len(features), -1)











# Le reste du code pour le prétraitement et l'entraînement reste inchangé
# ...
# ...

# Prétraiter les données pour l'entraînement du modèle
labels = np.array(labels)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)

# Diviser les données en ensembles d'entraînement et de test
train_features, test_features, train_labels, test_labels = train_test_split(flattened_features, labels_onehot, test_size=0.2, random_state=42)

# Construire le modèle de classification
classification_model = Sequential()
classification_model.add(Dense(512, activation='relu', input_dim=flattened_features.shape[1]))
classification_model.add(Dropout(0.5))
classification_model.add(Dense(len(label_encoder.classes_), activation='softmax'))

classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraîner le modèle de classification
classification_model.fit(train_features, train_labels, epochs=10, batch_size=32, validation_split=0.1)

# Évaluer le modèle de classification sur les données de test
classification_test_loss, classification_test_accuracy = classification_model.evaluate(test_features, test_labels)
print(f"Classification Test Accuracy: {classification_test_accuracy}")
# Définir le chemin complet où vous souhaitez enregistrer le modèle
save_path = '/home/deo/.config/spyder-py3/vraie images'

# Enregistrer le modèle avec le chemin complet
classification_model.save(os.path.join(save_path, 'nouveau depart_model.h5'))
print("Classification model saved at:", os.path.join(save_path, 'classification_model.h5'))
