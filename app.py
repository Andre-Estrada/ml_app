import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

s = pd.read_csv("social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1,1,0)
    return x

ss = pd.DataFrame({
    "sm_li":s["web1h"].apply(clean_sm),
    "income":np.where(s["income"] > 9, np.nan, s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] >= 8, np.nan, np.where(s["par"] == 1,1,0)),
    "married":np.where(s["marital"] >= 8, np.nan, np.where(s["marital"] == 1,1,0)),
    "female":np.where(s["gender"] > 3, np.nan, np.where(s["gender"] == 2,1,0)),
    "age":np.where(s["age"] > 98, np.nan, s["age"])
})

ss = ss.dropna()

y = ss["sm_li"]
x = ss[["income", "education", "parent", "married", "female", "age"]]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=123)

lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

st.title("Do they use LinkedIn? :female-office-worker::male-office-worker:")
inc_s = st.selectbox("Select Income Level", 
             options = ["Less than $10,000",
                        "10 to under $20,000",
                        "20 to under $30,000",
                        "30 to under $40,000",
                        "40 to under $50,000",
                        "50 to under $75,000",
                        "75 to under $100,000",
                        "100 to under $150,000",
                        "$150,000 or More"])

if inc_s == "Less than $10,000":
    inc_n = 1
elif inc_s == "10 to under $20,000":
    inc_n = 2
elif inc_s == "20 to under $30,000":
    inc_n = 3
elif inc_s == "30 to under $40,000":
    inc_n = 4
elif inc_s == "40 to under $50,000":
    inc_n = 5
elif inc_s == "50 to under $75,000":
    inc_n = 6
elif inc_s == "75 to under $100,000":
    inc_n = 7
elif inc_s == "100 to under $150,000":
    inc_n = 8  
else:
    inc_n = 9

#st.write(inc_n)

edu_s = st.selectbox("Select Education Level", 
             options = ["Less than High School",
                        "Some High School",
                        "High School Graduate",
                        "Some College",
                        "Two-Year Associate's Degree",
                        "Four-Year Bachelor's Degree",
                        "Some Graduate School",
                        "Postgraduate Degree"])

if edu_s == "Less than High School":
    edu_n = 1
elif edu_s == "Some High School":
    edu_n = 2
elif edu_s == "High School Graduate":
    edu_n = 3
elif edu_s == "Some College":
    edu_n = 4
elif edu_s == "Two-Year Associate's Degree":
    edu_n = 5
elif edu_s == "Four-Year Bachelor's Degree":
    edu_n = 6
elif edu_s == "Some Graduate School":
    edu_n = 7 
else:
    edu_n = 8

#st.write(edu_n)

par_s = st.select_slider("Parent?",["No","Yes"])

if par_s == "Yes":
    par_n = 1
else:
    par_n = 0

#st.write(par_n)

mar_s = st.select_slider("Married?",["No","Yes"])

if mar_s == "Yes":
    mar_n = 1
else:
    mar_n = 0

#st.write(mar_n)

gen_s = st.select_slider("Select Gender",["Male","Female"])

if gen_s == "Female":
    gen_n = 1
else:
    gen_n = 0

#st.write(gen_n)

age_n = st.slider("Select Age", 1, 97)

#st.write(age_n)


newdata = [inc_n, edu_n, par_n, mar_n, gen_n, age_n]
predicted_class = lr.predict([newdata])
proba = lr.predict_proba([newdata])

def pred():
    if predicted_class == 1:
        st.success("# This person uses LinkedIn :nerd_face:")
        st.write("### Probability of using LinkedIn:","{0:.0%}".format(proba[0][1]))
    else:
        st.error("# This person does not use LinkedIn :no_mobile_phones:")
        st.write("### Probability of using LinkedIn:","{0:.0%}".format(proba[0][1]))

st.button("Predict", on_click=pred)




st.write("This app was created by Andre Estrada")