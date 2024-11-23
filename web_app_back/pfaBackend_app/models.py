from django.db import models
import tensorflow as tf

# Chargez le modèle à partir du fichier HDF5
model_path = 'C:/Users/benzaied saif/Desktop/PFA_web/BackEnd_django/pfa_web_backend/pfaBackend_app/myModels/cnn_full_model.h5'
model = tf.keras.models.load_model(model_path)