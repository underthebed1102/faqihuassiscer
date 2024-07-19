import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from constants import *
from utils.ml_helper import one_hot_encode, binning_data
import data_visualization


# Global variables for UI
navigation_menus = ["Home", "Data Visualization"]
about_description = f"""
            ### About

            The app is based on Neural Networks that being trained by the National Health and Nutrition Examination Survey (NHANES)
            data. You can find how the data was preprocessed [here]({ML_REPO})
            and the app source code [here]({APP_REPO}) on GitHub.
        """
error_msg = f"""Oops, something went wrong. Please contact the developer at {DEV_EMAIL}"""
page_title = "Osteoporosis Prediction"
osteo_basics = """
            Osteoporosis, the most common bone disease, occurs when bone mineral density and bone mass decrease, 
            or there are changes in bone structure and strength. In 2010, an estimated 10.2 million people in the United States aged 50 
            and over had osteoporosis, and an estimated 43.3 million others had low bone mass. However, it is a silent disease that 
            most people with osteoporosis do not know they have it until they break a bone.
        """
instruction_prediction = """
            * Want to predict whether you have osteoporosis? Answer the following questions and click Predict.\n
              The value range of questions are from answers of the osteoporosis and its related questionnaires of NHANES data, where respondents were aged 40-80 (and plus).\n
              If there is no value in the options that exactly matches your answer, choose the closest value.\n
              Don't worry, your personal data will not be stored.
        """
instruction_visualization = """
            * Want to visualize the prevalence of osteoporosis? Select Data Visualization through the left menu 
        """
disclaimer_about_result = """
            Note: We consider confidence score greater than 50\% as having osteoporosis.\n
            
            Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. 
            Always consult with a qualified healthcare professional if you have any concerns about your health.
        """

input_widgets = {
    "Gender": "Gender",
    "Race": "Race",
    "Age": "Age",
    "Sleep Duration (Hours)": "How much sleep do you get (hours)?",
    "BMI": "Body Mass Index (BMI)",
}
radio_input_widgets = {
    "Smoking": "Have you smoked at least 100 cigarettes in life?",
    "Heavy Drinking": "Have you ever had a time in your life when you drank 4/5 or more alcoholic beverages almost every day?",
    "Arthritis": "Has a doctor or other health professional ever told you that you had arthritis?",
    "Liver Condition": "Has a doctor or other health professional ever told you that you had any kind of liver condition?",
    "Parental Osteoporosis": "Either of your parents ever told had osteoporosis or brittle bones?",
}

@st.cache_data(show_spinner=False)
def load_model():
    """Load the saved machine learning model"""

    try:
        model = pickle.load(open(MODEL_PICKLE, 'rb'))
        return model
    except FileNotFoundError:
        st.error(error_msg)
        st.stop()

@st.cache_data(show_spinner=False)
def load_example_data():
    """Load the example data (data used to train the model)"""

    try:
        data = pd.read_csv(EXAMPLE_DATA_FILE)
        return data
    except FileNotFoundError:
        st.error(error_msg)
        st.stop()   

@st.cache_data(show_spinner=False)
def load_mean_std():
    """Load the mean and standard deviation of the data used to train the model"""

    try:
        data = pd.read_csv(EXAMPLE_MEAN_STD_FILE)
        mean = data.iloc[0]
        std = data.iloc[1]
        return mean, std
    except FileNotFoundError:
        st.error(error_msg)
        st.stop()


@st.cache_data(show_spinner=False)
def load_css():
    """Load the custom styling file"""

    try:
        with open(CSS_FILE) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

@st.cache_data(show_spinner=False)
def load_banner():
    """Load the banner image"""

    try:
        img= Image.open(BANNER_IMG)
        st.image(img)
    except:
        pass

def encode_input_data(input_data, example_data):
    """Return the encoded user input data"""

    # Combine user input features with entire survey dataset for encoding the input features
    df_all = pd.concat([input_data, example_data], axis=0)
    # One hot encode
    df_all_encoded = one_hot_encode(df_all)
    # The first row is the encoded input data
    df_input_encoded = df_all_encoded[:1]

    return df_input_encoded

def preprocess_input_data(input_data, example_data, mean, std):
    """Return encoded, binned and normalized user input data"""

    # Encode the input data
    df_input_encoded = encode_input_data(input_data, example_data)
    # Binning
    df_input_encoded = binning_data(df_input_encoded) 
    # Standardize the input data
    df_input_scaled = (df_input_encoded - mean)/std

    return df_input_scaled

def create_selectbox(data, widget_key):
    """Create a select widget"""

    unique_values = data[widget_key].unique()
    widget_values = tuple(sorted(unique_values))

    return st.selectbox(label=input_widgets[widget_key], options=widget_values)

def create_number_input(data, widget_key, precision_num, step):
    """Create a numeric input widget"""

    if precision_num == 0:
        min_v = int(data[widget_key].min())
        mean_v = int(data[widget_key].mean())
        max_v = int(data[widget_key].max())
    else:
        min_v = round(data[widget_key].min(), precision_num).item()
        mean_v = round(data[widget_key].mean(), precision_num).item()
        max_v = round(data[widget_key].max(), precision_num).item()

    return st.number_input(label=input_widgets[widget_key], min_value=min_v, max_value=max_v, value=mean_v, step=step)

def user_input_features(example_data):
    """Create input widgets with default values and value range from `example_data`"""

    featureValues = dict()

    with st.container():
        col1, space, col2 = st.columns((10,1,10))
        with col1:
            widget_key = "Gender"
            featureValues[widget_key] = create_selectbox(example_data, widget_key)
        with col2:
            widget_key = "Race"
            featureValues[widget_key] = create_selectbox(example_data, widget_key)

    with st.container():
        col1, space, col2 = st.columns((10,1,10))
        with col1:
            widget_key = "Age"
            precision_num = 0
            step = 1
            featureValues[widget_key] = create_number_input(example_data, widget_key, precision_num, step)
        with col2:
            widget_key = "Sleep Duration (Hours)"
            precision_num = 1
            step = 0.5
            featureValues[widget_key] = create_number_input(example_data, widget_key, precision_num, step)

    with st.container():
        widget_key = "BMI"
        min_v = round(example_data[widget_key].min(), 1).item()
        max_v = round(example_data[widget_key].max(), 1).item()
        featureValues[widget_key] = st.slider(label=input_widgets[widget_key], min_value=min_v, max_value=max_v, value=min_v)    

    # Create radio input widgets
    for widget_key, widget_label in radio_input_widgets.items():
        with st.container():
            unique_values = example_data[widget_key].unique()
            widget_values = tuple(sorted(unique_values))
            featureValues[widget_key] = st.radio(label=widget_label, options=widget_values, horizontal=True)

    # Save input data       
    input_data = pd.DataFrame(featureValues, index=[0])

    return input_data

def get_prediction(model, input_data_preprocessed):
    """Get prediction results"""

    # Make predictions
    prediction_logic = model.predict(input_data_preprocessed)
    # Get the probability of 1 (with osteoporosis)
    tf_prediction = tf.nn.sigmoid(prediction_logic)
    prediction_prob = tf.get_static_value(tf_prediction)[0].item()
    # The predicted label will be 1 only when the probability > 0.5
    prediction_index = round(prediction_prob)
    existence = np.array(['No', 'Yes'])
    prediciton_str = existence[prediction_index]

    return prediciton_str, prediction_prob

def main():
    # Load styling
    load_css()

    # Load the cleaned example data (data of the National Health and Nutrition Examination Survey)
    example_data = load_example_data()
    # Get the mean and std of preprocessed example data
    # Prepare for standardization
    mean, std = load_mean_std()
    
    # Load the saved model
    if "model" not in st.session_state:
        st.session_state["model"] = load_model()

    choice = st.sidebar.selectbox("Menu", navigation_menus)

    # When in home page
    if choice == "Home":
        st.sidebar.markdown(about_description)
        st.title(page_title)

        # Banner image
        load_banner
