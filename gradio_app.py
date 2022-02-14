import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('game-or-book-cover-model.h5')

labels = ['Game', 'Book']

width = 130
height = 180

def classify_image(cover):
  normalized_img = cover / 255
  input = np.expand_dims(normalized_img, axis=0)
  prediction = model.predict(input)
  return {labels[i]: float(prediction[0][i]) for i in range(len(labels))}

cover = gr.inputs.Image(shape=(width, height), label='Upload cover image to classify')
label = gr.outputs.Label(label='Model prediction')

examples = ['fifa15.jpg', 'lotr.jpg', 'gta.jpg', 'sapiens.jpg', 'life3.jpg', 'fastai.jpg']

interface = gr.Interface(fn=classify_image, 
             inputs=cover, 
             outputs=label, 
             title="Game or book cover classifier",
             description="Classify if it's game or book cover with this neural network model created using Tensorflow library.", 
             theme="dark-grass", 
             examples=examples,
             allow_flagging="never")

interface.launch()
