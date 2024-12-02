import tensorflow as tf

pb_model = tf.keras.models.load_model('C:/potato_plant_disease_classification/potatoes_model/1.keras')

# pb_model.save('C:/potato_plant_disease_classification/potatoes_model/')

pb_model = pb_model.export('C:/potato_plant_disease_classification/potatoes_model')