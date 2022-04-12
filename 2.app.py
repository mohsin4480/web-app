from pyexpat import features
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

#Step#1 make containers
header=st.container()
data_sets=st.container()
features=st.container()
model_training=st.container()

with header:
    st.title("Kashti  ke app")
    st.text("we will make kashti dataset app")

with data_sets:
    st.header("kashti doob gae, Haw!")  
    st.text("we have titanic dataset")
     
    #import dataset
    df=sns.load_dataset("titanic")
    df=df.dropna()
    #check head 
    st.write(df.head(10))
    #bar chart of categorical var
    st.subheader("Ary o sambha kitne admi they")
    st.bar_chart(df["sex"].value_counts())
    #bar chart of cat var
    st.subheader("class k hisab se fark")
    st.bar_chart(df["class"].value_counts())
    #bat chart of numeric var
    st.bar_chart(df["age"].head(10))


with features:
    st.header("These are our app features")
    st.text("There are may features in titanic dataset")
    #markdown
    st.markdown("1. **Feature 1**: Age")
    st.markdown("2. **Feature 2**: Sex")
    st.markdown("3. **Feature 3**: Class")
    st.markdown("4. **Feature 4**:  Gender")

with model_training:
    st.header("kashti walon ka kya bna?")    
    st.text("we will build a model")

    #making columns
    input, display=st.columns(2)

    #pehle column mn selection points hon
    max_depth=input.slider("How many people do you know?", min_value=10,max_value=100,value=20,step=5) 

#n_estimators
n_estimators=input.selectbox("How many tree should be there in RF?", options=[50,100,200,300,"No limit"])

#adding list of features
input.write(df.columns)

#input features from users
input_features=input.text_input('which feature should we use?')

#Machine learning model
model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

#yahan pr hm aik condition lgayen ge
if n_estimators=='No limit':
    model=RandomForestRegressor(max_depth=max_depth)
else:
    model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

#define x and y
X=df[[input_features]]
y=df[["fare"]]

#fit our model
model.fit(X,y)
pred=model.predict(y)
    
#display metrices
display.subheader("mean absolute error is: ")
display.write(mean_absolute_error(y,pred))
display.subheader("mean squared error is: ")
display.write(mean_squared_error(y,pred))
display.subheader("r2  score error is: ")
display.write(r2_score(y,pred))