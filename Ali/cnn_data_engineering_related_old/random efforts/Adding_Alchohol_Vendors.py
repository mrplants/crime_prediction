

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






########################################################
#preparing for the Nearest Neighbor calculations 
#######################################################


result_grocery_loc = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data processed\grocery_joined___key_2018-11-14 11-49-09.xlsx')
result_grocery = getDF(loc = result_grocery_loc, sheetname = 'data')
result_simplified = result_grocery[['index_', 'Latitude','Longitude']].copy()
result_simplified.rename(columns = {'Latitude':'latitude','Longitude':'longitude'}, inplace = True)




liquor_store_loc = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime still need to explore\Business_Licenses_-_Current_Liquor_and_Public_Places_of_Amusement_Licenses.csv')
liquor_store_data = pd.read_csv(liquor_store_loc, sep=',', engine='python')

#clean up the liquor store data column that has "﻿ID" as a column
columns_of_data = liquor_store_data.columns.tolist()
foundedCol = None
for col in columns_of_data:
    if('�' in col):
        foundedCol = col 
        break 

if(type(foundedCol) != type(None)):
    print("found Id column")
    liquor_store_data.rename(columns ={foundedCol:'ID'}, inplace= True)



values2 = liquor_store_data[['ID', 'LATITUDE', 'LONGITUDE']].copy()
values2.rename(columns = {'LATITUDE':'latitude', "LONGITUDE":'longitude', 'ID':'category'}, inplace = True)

assert(len(liquor_store_data.ID.unique()) == len(liquor_store_data) )

values2=values2.dropna()

df = result_simplified
values = values2 
nameToCall = "key_liquorStores"




########################################################
#Nearest Neighbor calculations 
#######################################################


#Requirements:
#values is a dataframe and should have 3 columns [category, latitude, longitude]
#nameTocall -> should have no spaces and shouldn't be too long
#df -> must have an "index_" column, and a 'latitude' and 'longitude' column 
ctr = 0
def KNN_(df, values, nameToCall): #values should have 3 columns [category, latitude, longitude]
    
    values['index_'] = values.index
    
    writeDFToFile(dfs=[( values[['index_','category']].copy()  , nameToCall )] , path_ = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data lookups\lookup_'+nameToCall+"_" )
    
    arr = values[['latitude', 'longitude']].copy().values.T
    
    #column to write 
    global ctr 
    ctr = 0
    def functionToApply(row):
        point = np.array([[ row['latitude'] ],
                           [row['longitude']]])   
        dists = np.sum(((arr-point)**2), axis = 0)
        
        #print('--')
        global ctr 
        #print('---')
        ctr+=1
        if(ctr %10000 == 0):
            print(ctr)

        return np.argmax(dists.tolist())
    
    t1 = datetime.datetime.now()
    print("Starting at ", t1)


    df['closestc'] = df.apply(functionToApply, axis = 1)


    t2 = datetime.datetime.now()
    print("Ending at ", t2)

#Run KNN_ function
df = KNN_(df, values, nameToCall)


########################################################
#Saving NN calculations 
#######################################################

# have df , need to join in the result 
df = df[['index_', 'closestc']].copy()
df.rename(columns = {'closestc':'closestLiquorStore'}, inplace =True)
printNulls(df)
printNulls(result_grocery)
result_grocery_liquor = pd.merge(left = result_grocery, right = df, on = 'index_', how = 'left')

writeDFToFile(dfs=[( result_grocery_liquor , 'data' )] , path_ = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data processed\grocery_liquor_joined___'+nameToCall+"_" )


