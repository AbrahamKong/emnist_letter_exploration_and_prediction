import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import pandas as pd
import plotly.express as px


# Cache model to load faster an only do it once rather than on every refresh
@st.cache(allow_output_mutation = True)
def load_models():
  # Load in the pre-trained model
  model_file_path = 'https://rawcdn.githack.com/AbrahamKong/emnist_letter_exploration_and_prediction/814b4bf874284ea02aeddd213d3f1e5b0925af0a/models/model_capital_letters_v01.h5'
  model = tf.keras.models.load_model(model_file_path)
  return model

# Load the models
model = load_models()

# Define some valiables and helper functions
labels_caps = [
               'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
              ]
labels_dict_caps = {key: value for key, value in enumerate(labels_caps)}

# Define the Plotyly function
def create_probability_fig(model_prediction):
    df_temp = pd.DataFrame(model_prediction, columns=labels_dict_caps.values())
    df_temp = df_temp.transpose().reset_index()
    df_temp.columns = ['Label','Probability']
    fig = px.bar(df_temp, x='Label', y='Probability')
    return fig


# Create the Application
st.title('Letter and Didgit Prediction')

# Create a 2 column Streamlit layout
col1, col2 = st.columns(2)

with col1:
  st.markdown('Draw a capital letter here:')
  # Create a drawing canvas with desired properties
  canvas_result = st_canvas(
      fill_color="#ffffff",
      stroke_width=10,
      stroke_color='#ffffff',
      background_color="#000000",
      height=200,
      width=200,
      drawing_mode='freedraw',
      key="canvas",
  )

with col2:
  # Show that the resized image looks like
  st.markdown("What the model see's as input:")
  if canvas_result.image_data is not None:
      img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
      img_rescaling = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
      st.image(img_rescaling)

# Generate the prediciton based on the users drawings
if st.button('Predict'):
    x_user_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = model.predict(x_user_input.reshape(1, 28, 28, 1))
    pred_label = labels_dict_caps[pred.argmax()]
    st.write(f'Predicted Label: {pred_label}')

    # Create a plotly barchart of the predicted probablities
    fig = create_probability_fig(pred)
    st.plotly_chart(fig, use_container_width = True)
