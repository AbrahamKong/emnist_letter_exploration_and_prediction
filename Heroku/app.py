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
  model_file_path = 'reference/model_conv_v03.h5'
  model = tf.keras.models.load_model(model_file_path)
  return model

# Load the models
model = load_models()

# Define some variables and helper functions
labels = [
            # '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'a', 'b', 'd', 'e', 'f','g', 'h', 'n', 'q', 'r', 't',
          ]
labels_dict = {key: value for key, value in enumerate(labels)}

# Define the Plotly function
def create_probability_fig(model_prediction):
    df_temp = pd.DataFrame(model_prediction, columns=labels_dict.values())
    df_temp = df_temp.transpose().reset_index()
    df_temp.columns = ['Label','Probability']
    fig = px.bar(df_temp, x='Label', y='Probability')
    return fig


# Create the Application
st.title('Letter Prediction')

# Create a 2 column Streamlit layout
col1, col2 = st.columns(2)

with col1:
  st.markdown('Draw a letter here:')
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

# Generate the prediction based on the users drawings
if st.button('Predict'):
    x_user_input = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x_user_input = x_user_input.reshape(1, 28, 28, 1) / 255 # Reshape and normalize data
    pred = model.predict(x_user_input)
    pred_label = labels_dict[pred.argmax()]
    st.header(f'Predicted Label: {pred_label}')

    # Create a Plotly barchart of the predicted probabilities
    fig = create_probability_fig(pred)
    st.plotly_chart(fig, use_container_width = True)


# Show example predictions images
st.header('Example Predictions')
st.image('reference/letter_predictions_img_196.png')