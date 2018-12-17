

import sys

import pandas as pd
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt

import pathlib
import datetime
_="""

Define any useful functions 

"""

def cleanURL(url):
    p = pathlib.Path(url)
    path = str(p.as_posix()) 
    return path 


def getDF(loc, sheetname):
    dataframe = pd.read_excel(loc, sheetname)
    #https://stackoverflow.com/questions/40950310/strip-trim-all-strings-of-a-dataframe
    dataframe = dataframe.applymap(lambda x: x.strip() if type(x) is str else x)
    if('WCCT_DESC' in dataframe.columns.tolist()):
        dataframe['WCCT_DESC'] = dataframe['WCCT_DESC'].str.strip()

    if('WCTR_CD' in dataframe.columns.tolist()):
        dataframe['WCTR_CD'] = dataframe['WCTR_CD'].str.strip()

    if('ORDR_PART_NO' in dataframe.columns.tolist()):
        dataframe['ORDR_PART_NO'] = dataframe['ORDR_PART_NO'].str.strip()
    return dataframe

def printNulls(df):
    null_columns = df.columns[df.isnull().any()]
    return df[null_columns].isnull().sum() 


def writeDFToFile(dfs, path_): #dfs is an array of dataframes and their sheet names , path needs to have
    time_ = str(datetime.datetime.now())
    current_date_time = time_[0:time_.index(".")]
    current_date_time = current_date_time.replace(":", "-")
    task4_fileoutput = path_+current_date_time+".xlsx"

    writer = pd.ExcelWriter(task4_fileoutput)
    
    for df_tuple in dfs:  
        df = df_tuple[0]
        sheetName = df_tuple[1]
        df.to_excel(writer, sheetName)
    print("file written to :       " + task4_fileoutput)
    writer.save()

_="""
Start with analysis, get the crime data 
"""



print("started")
#change location here 
path = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\Crimes_-_2001_to_present.csv'
path1 = path.replace('\\', r'//')
crime_rate = pd.read_csv(path1, sep=',', engine='python')
print("done")
crime_rate = crime_rate.dropna()

#change the Primary Type - the thing we want to predict - to discrete category numbers 
crime_rate['categoryType'] =  pd.Categorical(crime_rate['Primary Type'])
crime_rate['categoryCode'] = crime_rate['categoryType'].cat.codes            # df.cc.astype('category').cat.codes  https://stackoverflow.com/questions/38088652/pandas-convert-categories-to-numbers

crime_rate['time_'] = pd.to_datetime(crime_rate['Date'], format ="%m/%d/%Y %I:%M:%S %p")
crime_rate['hr'] = crime_rate.time_.dt.hour
crime_rate['min'] = crime_rate.time_.dt.minute
# the month and year 
crime_rate['month_'] = crime_rate.time_.dt.month   #modified on 11/10 after talking to Sean 
crime_rate['year_'] = crime_rate.time_.dt.year

bk2= crime_rate.copy()


###get the category codes -> primary type vs other 
category_codes = bk2.drop_duplicates(subset = ['Primary Type'], keep = 'first')  
category_codes_ = category_codes[['Primary Type', 'categoryCode']].copy()



output = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\category_codes_')
writeDFToFile(dfs=[(category_codes_, 'category_codes')], path_ = output )







##################################################
#Below is code for making the cnn data the backend supplies to the frontend
##################################################



## make a new dataframe for the data
#need the categoryCode, lat, long, week, year
#we need to add a color type 
#images are going to be stored as categoryCode_week_year 
output2 = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\cnn_data__')

crime_rate['week_'] = crime_rate.time_.dt.week
#the minumum week is 1 and maximum week is 53
#maximum year is 2018, and minimum year is 2001
#https://stackoverflow.com/questions/35268817/unique-combinations-of-values-in-selected-columns-in-pandas-data-frame-and-count
year_weeks = crime_rate.groupby(['year_', 'week_']).size().reset_index().rename(columns= {0:'count'})

#so ideally we should have (2018 - 2001 +1)*53 weeks= 954 rows in year_weeks. 
#we only have 932. Which means our dataset ends somewhere before the end of 2018 (we are 22 rows short) 
#let's sort on year and then week: 
year_weeks.sort(columns = ['year_', 'week_'], ascending = [1,1], inplace = True)
#looking at year_weeks.tail() says we end at week 42 of 2018. -> meaning we should only be about 10 weeks short of (week53,year 2018). Some weeks, no crime was reported
#in short we will have no data for those weeks and we can add a 35th category (categoryCode = 34) saying there is no 

crime_rate_cnn_data = crime_rate[['year_', 'week_', 'categoryCode', 'Latitude', 'Longitude']].copy()

output2 = output2+"11_25_2018.csv"
crime_rate_cnn_data.to_csv(output2, sep=',')  # written to 'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/cnn_data__11_25_2018.csv'
#writeDFToFile(dfs=[(crime_rate_cnn_data, 'cnn_data')], path_ = output2 )  

#to check if all possible weeks/years are included 

#the most populace data points 
abc = crime_rate_cnn_data.groupby(['year_', 'week_', 'categoryCode']).size().reset_index().rename(columns = {0:'count'}).sort(['count'], ascending = [0], inplace = False)




