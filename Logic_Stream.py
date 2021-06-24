import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Classification.csv')
x = df.iloc[:, :-1].values    #Attributes,Hours(indepedent) 
y = df.iloc[:, 1].values      # Labels , Score (Depend)
model = LogisticRegression()
model = model.fit(x, y)

import streamlit

def lr_prediction(var_1):
     array=np.array([var_1])
     a=array.reshape(-1, 1)
     if (model.predict(a)==0):
          return "Fail"
     else:
          return "Pass"
            
    
def run():
     streamlit.title("Students Score prediction Model")
     html_temp="""
     """
     streamlit.markdown(html_temp)
     Hours=streamlit.text_input("Hours Studied by student")
     prediction=""
     if streamlit.button("Predict"):
          prediction=lr_prediction(Hours)

     streamlit.write(prediction)

if __name__=='__main__':
     run()



