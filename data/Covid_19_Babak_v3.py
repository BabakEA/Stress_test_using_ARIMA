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
    """ input  : Json Url ,

        Output : Json Data  """
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
        """Creat a data fram from the first Country in the list""" 
        DF=json_normalize(self.data[self.Country_list[0]])
        DF.set_index(["date"], inplace = True, 
                 append = False, drop = True)
        
        Col=[self.Country_list[0],self.Country_list[0]+"_D",self.Country_list[0]+"_R"]
        DF.columns=Col
        DF.index=pd.to_datetime(DF.index)
        self.DF=DF

        
    def Json_to_Pandas_kids(self):
        """ append the rest for the countries to the
                Pandas Datafram and updat self.DF """
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
        #Country_list=self.Country_list
        
        self.dropdown_class =widgets.Dropdown(options =
                                 ['ALL','EXIT']+ sorted(self.Country_list), description= "Select Country: " )
        self.flag="All"

        self.output = widgets.Output()
        
        self.dropdown_class.observe(self.dropdown_class_eventhandler, names='value')
        
        
        display(self.dropdown_class)
        display(self.output)
        
    def dropdown_class_eventhandler(self,name):

        self.change=self.dropdown_class.value

        self.output.clear_output()

        with self.output:

            if (self.change== self.ALL):
                display(self.DF)
                
            elif(self.change=="EXIT" ):
                print("Thnak you for using my Library \n Cheers \n Babak")
                

            else:
                Col=[x for x in self.DF.columns.tolist() if self.change in x]
                self.df=self.DF[Col]
                Infected=self.df[Col[0]].tolist()
                Infected=[0]+Infected
                self.df["Daily_infected"]=[y - x if y > x else 0 for x,y in zip(Infected,Infected[1:])]
                
                #Country_select.ARIMA_Death_Ratio(self.DF[Col])
                ARIMA_COL=[Col[0]]+['Daily_infected']
                Title_list=["Totla Infected in_"+ARIMA_COL[0] ,ARIMA_COL[1]+"_"+ARIMA_COL[0]]
                
                self.COHORT_report(Col[0]) 
                for i in range(0,2):
                    self.ARIMA_Death_Ratio(ARIMA_COL[i],Title_list[i])
                  
                #self.Cohort_report(Col[0])
                #self.ARIMA_Death_Ratio()
                #self.ARIMA_Death_Ratio("Daily_infected",)
                
                #display(df)
    def report_old(self):
        x=self.x
        y=self.y
        hue=self.hue
        style=self.style
        data=self.report_df
        plt.figure(figsize=(13,10))
        
        ax = sns.lineplot(x="Day", y=col[0],
                  hue="Class", style="Class",
                  markers=True, dashes=True, data=DF_SNS)
        ax = plt.gca()
        ax.set_xticklabels(DF_SNS['Day'],rotation=45,fontsize='small')
        ax.set(xlabel=col[0]+"__Daily Report", ylabel= "Number of Infected in "+col[0])
        plt.title("ARIMA Stress Test")
        plt.show()        

    def ARIMA_Death_Ratio(self,col,sTR_report):        

        df=self.df

        train=df[0:-5] 
        test=df[-6:]
        
        model=auto_arima(train[col], start_p=1, start_q=1,
                           max_p=30, max_q=30, m=10,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
        print(model.aic())
        model.fit(train[col])
        forcast = model.predict(n_periods=len(test[col]))

        train["Class"]="Training"
        test["Class"]="Valid Data"
        ARIMA_FORCAST=test.copy()
        ARIMA_FORCAST["Class"]="Forcast"
        ARIMA_FORCAST[col]=forcast
        
        DF_SNS=pd.concat([train,test,ARIMA_FORCAST])
        del (test,train,ARIMA_FORCAST)
        DF_SNS['Day']=DF_SNS.index
        DF_SNS['Day'] = DF_SNS['Day'].map(lambda x: x.strftime('%Y-%m-%d'))
        #DF_SNS['Day'] =  DF_SNS['Day'].astype(str)
        DF_SNS[col] = DF_SNS[col].astype(np.float64)

        plt.figure(figsize=(13,10))
        
        ax = sns.lineplot(x="Day", y=col,
                  hue="Class", style="Class",
                  markers=True, dashes=True, data=DF_SNS)

        ax = plt.gca()

        ax.set_xticklabels(DF_SNS['Day'],rotation=45,fontsize='small')
        ax.set(xlabel="__Daily Report__", ylabel= sTR_report)

        plt.title("ARIMA Stress Test for  {}".format(sTR_report))
       
        plt.show()
        #del DF_SNS
        self.DF_SNS=DF_SNS
        
    def COHORT_report(self,col):
        infected_list=self.df[col].tolist()
        tem_index=self.df.index.map(lambda x: x.strftime('%Y-%m-%d')).tolist()
        max_size=len(infected_list)
        data=[]
        for i in range(0,max_size):

            tem_list=[y - x if y > x else 0 for x,y in zip(infected_list[i:],infected_list[i+1:])]
            tem_list = (tem_list + [np.nan] * max_size)[:max_size-1]
            data.append(tem_list)


        self.COHORT_DF=pd.DataFrame(data,index=tem_index) 
        del data
            
    def Cohort_plt(self,df):
        plt.figure(figsize = (30,30))
        plt.title('Cohort Analysis - residual daily infected')
        sns.heatmap(data = df, 
                    annot = True, 
                    cmap = "BuGn")
        plt.show()
    def help(self):
        print(help(Covid))

        print(" *********** ")

        print(help(Country_select))



            
            
            

        

        






