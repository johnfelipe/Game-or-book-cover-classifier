import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('game-or-book-cover-model.h5')

labels = ['Juego', 'Libro']

width = 130
height = 180

def classify_image(cover):
  normalized_img = cover / 255
  input = np.expand_dims(normalized_img, axis=0)
  prediction = model.predict(input)
  return {labels[i]: float(prediction[0][i]) for i in range(len(labels))}

cover = gr.inputs.Image(shape=(width, height), label='Cargar imagen de portada para clasificar')
label = gr.outputs.Label(label='Predicción del modelo')

examples = ['examples/fifa15.jpg', 'examples/lotr.jpg', 'examples/gta.jpg', 'examples/sapiens.jpg', 'examples/life3.jpg', 'examples/fastai.jpg']

interface = gr.Interface(fn=classify_image, 
             inputs=cover, 
             outputs=label, 
             title="Clasificador de cubierta de juego o libro",
             description="Clasificar si se trata de una cubierta de juego o libro con este modelo de red neuronal creado utilizando la biblioteca Tensorflow.", 
             theme="dark-grass", 
             examples=examples,
             allow_flagging="never")

interface.launch(server_name="0.0.0.0")
