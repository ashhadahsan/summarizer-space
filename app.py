import streamlit as st
import pandas as pd
from transformers import pipeline
from stqdm import stqdm
from simplet5 import SimpleT5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer
from tensorflow.keras.models import load_model
from tensorflow.nn import softmax
import numpy as np
from datetime import datetime
import logging
from constants import sub_themes_dict

date = datetime.now().strftime(r"%Y-%m-%d")
model_classes = {
    0: "Ads",
    1: "Apps",
    2: "Battery",
    3: "Charging",
    4: "Delivery",
    5: "Display",
    6: "FOS",
    7: "HW",
    8: "Order",
    9: "Refurb",
    10: "SD",
    11: "Setup",
    12: "Unknown",
    13: "WiFi",
}


@st.cache_resource
def load_t5():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    st.success("Loaded t5-base model!")
    return model, tokenizer


@st.cache_resource
def custom_model():
    st.success("Loaded custom summarization model!")
    return pipeline("summarization", model="my_awesome_sum/")


@st.cache_resource
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode("utf-8")


@st.cache_resource
def load_one_line_summarizer(model):
    st.success("Loaded one-line-summary model!")
    return model.load_model("t5", "snrspeaks/t5-one-line-summary")


@st.cache_resource
def classify_category():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    new_model = load_model("model")
    st.success("Loaded theme classification model!")
    return tokenizer, new_model

@st.cache_resource
def classify_sub_theme():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    new_model = load_model("sub_theme_model")
    st.success("Loaded sub-theme-model")
    return tokenizer, new_model


st.set_page_config(layout="wide", page_title="Amazon Review Summarizer")
st.title("Amazon Review Summarizer")

uploaded_file = st.file_uploader("Choose a file", type=["xlsx", "xls", "csv"])
summarizer_option = st.selectbox(
    "Select Summarizer",
    ("Custom trained on the dataset", "t5-base", "t5-one-line-summary"),
)
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    summary_yes = st.checkbox("Summrization", value=False)

with col2:
    classification = st.checkbox("Classify Category", value=True)

with col3:
    sub_theme = st.checkbox("Sub theme classification", value=True)

ps = st.empty()

if st.button("Process", type="primary"):
    cancel_button = st.empty()
    cancel_button2 = st.empty()
    cancel_button3 = st.empty()
    if uploaded_file is not None:
        if uploaded_file.name.split(".")[-1] in ["xls", "xlsx"]:

            df = pd.read_excel(uploaded_file, engine="openpyxl")
        if uploaded_file.name.split(".")[-1] in [".csv"]:
            df = pd.read_csv(uploaded_file)
        columns = df.columns.values.tolist()
        columns = [x.lower() for x in columns]
        df.columns = columns
        print(summarizer_option)
        output = pd.DataFrame()
        try:
            text = df["text"].values.tolist()
            output["text"] = text
            if summarizer_option == "Custom trained on the dataset":
                if summary_yes:
                    model = custom_model()

                    progress_text = "Summarization in progress. Please wait."
                    summary = []

                    for x in stqdm(range(len(text))):

                        if cancel_button.button("Cancel", key=x):
                            del model
                            break
                        try:
                            summary.append(
                                model(
                                    f"summarize: {text[x]}",
                                    max_length=50,
                                    early_stopping=True,
                                )[0]["summary_text"]
                            )
                        except:
                            pass
                    output["summary"] = summary
                    del model
                if classification:
                    classification_token, classification_model = classify_category()
                    tf_batch = classification_token(
                        text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="tf",
                    )
                    with st.spinner(text="identifying theme"):
                        tf_outputs = classification_model(tf_batch)
                    classes = []
                    with st.spinner(text="creating output file"):
                        for x in stqdm(range(len(text))):
                            tf_o = softmax(tf_outputs["logits"][x], axis=-1)
                            label = np.argmax(tf_o, axis=0)
                            keys = model_classes
                            classes.append(keys.get(label))
                        output["category"] = classes
                    del classification_token, classification_model
                if sub_theme:
                    classification_token, classification_model = classify_sub_theme()
                    tf_batch = classification_token(
                        text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="tf",
                    )
                    with st.spinner(text="identifying sub theme"):
                        tf_outputs = classification_model(tf_batch)
                    classes = []
                    with st.spinner(text="creating output file"):
                        for x in stqdm(range(len(text))):
                            tf_o = softmax(tf_outputs["logits"][x], axis=-1)
                            label = np.argmax(tf_o, axis=0)
                            keys = sub_themes_dict
                            classes.append(keys.get(label))
                        output["sub theme"] = classes
                    del classification_token, classification_model

                csv = convert_df(output)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"{summarizer_option}_{date}_df.csv",
                    mime="text/csv",
                )
            if summarizer_option == "t5-base":
                if summary_yes:
                    model, tokenizer = load_t5()
                    summary = []
                    for x in stqdm(range(len(text))):

                        if cancel_button2.button("Cancel", key=x):
                            del model, tokenizer
                            break
                        tokens_input = tokenizer.encode(
                            "summarize: " + text[x],
                            return_tensors="pt",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                        )
                        summary_ids = model.generate(
                            tokens_input,
                            min_length=80,
                            max_length=150,
                            length_penalty=20,
                            num_beams=2,
                        )
                        summary_gen = tokenizer.decode(
                            summary_ids[0], skip_special_tokens=True
                        )
                        summary.append(summary_gen)
                    del model, tokenizer
                    output["summary"] = summary

                if classification:
                    classification_token, classification_model = classify_category()
                    tf_batch = classification_token(
                        text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="tf",
                    )
                    with st.spinner(text="identifying theme"):
                        tf_outputs = classification_model(tf_batch)
                    classes = []
                    with st.spinner(text="creating output file"):
                        for x in stqdm(range(len(text))):
                            tf_o = softmax(tf_outputs["logits"][x], axis=-1)
                            label = np.argmax(tf_o, axis=0)
                            keys = model_classes
                            classes.append(keys.get(label))
                        output["category"] = classes
                    del classification_token, classification_model
                if sub_theme:
                    classification_token, classification_model = classify_sub_theme()
                    tf_batch = classification_token(
                        text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="tf",
                    )
                    with st.spinner(text="identifying sub theme"):
                        tf_outputs = classification_model(tf_batch)
                    classes = []
                    with st.spinner(text="creating output file"):
                        for x in stqdm(range(len(text))):
                            tf_o = softmax(tf_outputs["logits"][x], axis=-1)
                            label = np.argmax(tf_o, axis=0)
                            keys = sub_themes_dict
                            classes.append(keys.get(label))
                        output["sub theme"] = classes
                    del classification_token, classification_model
                csv = convert_df(output)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"{summarizer_option}_{date}_df.csv",
                    mime="text/csv",
                )

            if summarizer_option == "t5-one-line-summary":
                if summary_yes:
                    model = SimpleT5()
                    load_one_line_summarizer(model=model)

                    summary = []
                    for x in stqdm(range(len(text))):
                        if cancel_button3.button("Cancel", key=x):
                            del model
                            break
                        try:
                            summary.append(model.predict(text[x])[0])
                        except:
                            pass
                    output["summary"] = summary
                    del model

                if classification:
                    classification_token, classification_model = classify_category()
                    tf_batch = classification_token(
                        text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="tf",
                    )
                    with st.spinner(text="identifying theme"):
                        tf_outputs = classification_model(tf_batch)
                    classes = []
                    with st.spinner(text="creating output file"):
                        for x in stqdm(range(len(text))):
                            tf_o = softmax(tf_outputs["logits"][x], axis=-1)
                            label = np.argmax(tf_o, axis=0)
                            keys = model_classes
                            classes.append(keys.get(label))
                        output["category"] = classes
                    del classification_token, classification_model
                if sub_theme:
                    classification_token, classification_model = classify_sub_theme()
                    tf_batch = classification_token(
                        text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="tf",
                    )
                    with st.spinner(text="identifying sub theme"):
                        tf_outputs = classification_model(tf_batch)
                    classes = []
                    with st.spinner(text="creating output file"):
                        for x in stqdm(range(len(text))):
                            tf_o = softmax(tf_outputs["logits"][x], axis=-1)
                            label = np.argmax(tf_o, axis=0)
                            keys = sub_themes_dict
                            classes.append(keys.get(label))
                        output["sub theme"] = classes
                    del classification_token, classification_model

                csv = convert_df(output)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name=f"{summarizer_option}_{date}_df.csv",
                    mime="text/csv",
                )

        except KeyError:
            st.error(
                "Please Make sure that your data must have a column named text",
                icon="🚨",
            )
            st.info("Text column must have amazon reviews", icon="ℹ️")
        except BaseException as e:
            logging.exception("An exception was occurred")
