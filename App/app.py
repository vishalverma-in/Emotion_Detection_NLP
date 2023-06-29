# Core Packages
import streamlit as st
import altair as alt

# EDA Packages
import pandas as pd
import numpy as np

# Utils
import joblib
pipe_lr = joblib.load(open("models/emotion_classifier_pipe.pkl","rb"))

# Functions
def predict_emotions(docx):
    result = pipe_lr.predict([docx]);
    return result[0];

def predict_proba(docx):
    result = pipe_lr.predict_proba([docx])
    return result;


emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Main Application
def main():
    st.title("Emotion CLassifier App")
    menu = ["Home", "Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type here")
            submit_text = st.form_submit_button(label="Submit")
        
        if submit_text:
            col1, col2 = st.columns(2)
            pred_value = predict_emotions(raw_text)
            pred_proba = predict_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                pred_emoji = emotions_emoji_dict[pred_value]
                st.write("{}:{}".format(pred_value,pred_emoji))
                st.write("Confidence: {:.2f}".format(np.max(pred_proba)))

            with col2:
                st.success("Prediction Probabilitty")
                # st.write(pred_proba)
                proba_df = pd.DataFrame(pred_proba,columns=pipe_lr.classes_) 
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotions', y='Probability',color='Emotions')
                st.altair_chart(fig, use_container_width=True)




    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")
if __name__ == '__main__':
	main()