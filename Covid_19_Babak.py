"""

A simple Stress test on Covid Dataset using ARIMA 
Author : Babak.EA 
Date   : 2020-03-20, Persian New year 
Target : ARIMA stress test on infected population per Country 

"""

###############################

from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pandas.io.json import json_normalize 
import numpy as np
import statsmodels
import datetime
import urllib, json
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pmdarima.arima import auto_arima
import sys
import warnings
import ipywidgets as widgets
from IPython.display import display


if not sys.warnoptions:
    warnings.simplefilter("ignore")


#################################
####### Database : Json #########
URL="https://pomber.github.io/covid19/timeseries.json"

#################################

class Covid:
    def __init__(self,URL):
        self.URL=URL

        self.Load_Json()
        self.Json_to_Pandas()
        self.Json_to_Pandas_kids()

    def Load_Json(self):# input Json URL, Output: Json Data
        json_url =urllib.request.urlopen(self.URL)
        self.data = json.loads(json_url.read())
        self.Country_list=list(self.data.keys())
       
    def Json_to_Pandas(self): # Json to Pandas DataFrame
        DF=json_normalize(self.data[self.Country_list[0]])
        DF.set_index(["date"], inplace = True, 
                 append = False, drop = True)
        
        Col=[self.Country_list[0],self.Country_list[0]+"_D",self.Country_list[0]+"_R"]
        DF.columns=Col
        DF.index=pd.to_datetime(DF.index)
        self.DF=DF

        
    def Json_to_Pandas_kids(self):
        for i in self.Country_list[1:]:
            df_tem=json_normalize(self.data[i])
            df_tem.set_index(["date"], inplace = True, 
                     append = False, drop = True)
            Col=[i,i+"_D",i+"_R"]
            df_tem.columns=Col
            self.DF = pd.concat([self.DF, df_tem], axis=1)



class Country_select(Covid):
         
    def __init__(self):
        Covid.__init__(self,URL)
        self.ALL='ALL'
        
        self.dropdown_class =widgets.Dropdown(options =
                                 sorted(['ALL','EXIT']+ self.Country_list), description= "Select Country: " )
        self.flag="All"

        self.output = widgets.Output()
        
        self.dropdown_class.observe(self.dropdown_class_eventhandler, names='value')
        
        display(self.dropdown_class)
        display(self.output)

    """
    def unique_sorted_values_plus_ALL(self,array):
        self.array=array
        
        unique = self.array.unique().tolist()
        unique.sort()
        unique.insert(0, self.ALL)
        self.unique=unique
    """

    def dropdown_class_eventhandler(self,name):
        #print(name)
        
        #print(self.dropdown_class.value)
        self.change=self.dropdown_class.value

        self.output.clear_output()
        #Col=
        with self.output:

            if (self.change== self.ALL):
                display(self.DF)
                
            elif(self.change=="EXIT" ):
                print("Thnak you for using my Library")
                
                
            else:
                Col=[x for x in self.DF.columns.tolist() if self.change in x]
                self.df=self.DF[Col]
                Country_select.ARIMA_Death_Ratio(self.DF[Col])
                
                #display(df)
    def ARIMA_Death_Ratio(df):

        col=df.columns.tolist()
        train=df[0:-5] 
        test=df[-6:]        
        model=auto_arima(train[col[0]], start_p=1, start_q=1,
                           max_p=30, max_q=30, m=10,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        print(model.aic())

        model.fit(train[col[0]])

        forcast = model.predict(n_periods=len(test[col[0]]))
        forcast = pd.DataFrame(forcast,index = test.index,columns=['Prediction'])

        train["Class"]="Training"
        test["Class"]="Valid Data"
        forcast[col[1]]=0
        forcast[col[2]]=0
        forcast["Class"]="Forcast"
        forcast.columns=train.columns.tolist()        
        DF_SNS=pd.concat([train,test,forcast])
        DF_SNS['Day']=DF_SNS.index
        DF_SNS[col[0]] = DF_SNS[col[0]].astype(np.float64)

        ax = sns.lineplot(x="Day", y=col[0],
                  hue="Class", style="Class",
                  markers=True, dashes=False, data=DF_SNS)

        ax.set_xticklabels(DF_SNS['Day'],rotation=45)
        ax.set(xlabel=col[0]+"__Daily Report", ylabel= "Number of Infected in "+col[0])

       

        plt.show()
       

        
  
    


    
        






