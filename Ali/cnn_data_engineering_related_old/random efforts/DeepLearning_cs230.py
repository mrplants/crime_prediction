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
#bk = crime_rate.copy()

_="""
Explore anything about the dataset 
"""


# crime_rate.Date.min()
# crime_rate[date]= pd.to_datetime(crime_rate['Date'])
# crime_rate.dtypes
# crime_rate.head()
# len(crime_rate)
# crime_rate.columns.tolist()


#https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2   ---> 
# print('done')


_="""
After looking at the dataset, we see that there are Precincts we can join on , there are Police Districts we can join on and Community area as well 
"""


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


_="""
Join in the socio economic data -- columns to join on are "Community Area" and "Community Area Number"
"""

path = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime still need to explore\Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv'
path2 = path.replace('\\', r'//')
socioeconomic_data = pd.read_csv(path2, sep=',', engine='python')
socioeconomic_data = socioeconomic_data.dropna()  # there are currently 77 community areas: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2 
scbk = socioeconomic_data.copy()

#assert that the community area columns have the same datatypes otherwise we get null values 
#use: socioeconomic_data.dtypes 
assert(socioeconomic_data['Community Area Number'].dtype == crime_rate['Community Area'].dtype)
#rename one of the df's column to the other one's, see below -> so set Community Area Number to just Community Area so the columns match 
socioeconomic_data.rename(columns = {'Community Area Number':'Community Area'}, inplace = True )
#do the join and save
crime_socio = pd.merge(left = socioeconomic_data, right = crime_rate, on = 'Community Area', how = 'right')

printNulls(crime_socio) # shows you which columns have null values only 67 values, just delete them 
crime_socio = crime_socio.dropna()
cs_bk = crime_socio.copy() # back up the data 

#done with joining the socio economic data 

_="""
Join in the Liquor Store information -- columns to join on are 'POLICE DISTRICT'   and 'District'

"""

path = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime still need to explore\Business_Licenses_-_Current_Liquor_and_Public_Places_of_Amusement_Licenses.csv'
path3 = path.replace('\\', r'//')
liquor_store_data = pd.read_csv(path3, sep=',', engine='python')


columns_of_data = liquor_store_data.columns.tolist()
foundedCol = None
for col in columns_of_data:
    if('¿' in col):
        foundedCol = col 
        break 

if(type(foundedCol) != type(None)):
    print("found Id column")
    liquor_store_data.rename(columns ={foundedCol:'ID'}, inplace= True)


lsdbk = liquor_store_data.copy() 

# data exploration 
liquor_store_data.columns

#for now just join on the Police District and Distict , count the number of liqour stores in each distrcit
#NOTE:  that this is not the best thing to do -> ideally for each crime find the store that was closest to it -> this would be much more informational 
# problem is that there are 6644 unique stores : len(liquor_store_data['﻿ID'].unique().tolist()) and we have 6 million crimes -> that is computationally untractable. 
#might be able to find better ways of doing it though -> maybe make a dictionary for faster look ups 

#turns out that the lenght of liquor_store_data is the number of unique Id's <- this is great , the below works (no need to drop duplicates) 
liquor_store_data = liquor_store_data[['POLICE DISTRICT', 'ID']].copy()
liquor_store_data = liquor_store_data.dropna()  
len(liquor_store_data) # this is now less : 6602 rows , so had some nulls

stores_per_police_district= liquor_store_data.groupby(['POLICE DISTRICT']).count().reset_index()
#make a base measure table
# found at : most current list is https://data.cityofchicago.org/Public-Safety/Boundaries-Police-Districts-current-/fthy-xz3r ,
# turns out other police districts might have existed between 2001 to 2017 , therefore we can join and makea base measure table 

#stores per police has 22 different police districts
# crime_socio has 24 different police districts -> joining will cause problems -> so we can just remove the na values for now KLUDGE
stores_per_police_district.rename(columns = {'POLICE DISTRICT':'District', 'ID':'LiquorStoreCount_District'} , inplace = True )

#assert same datatypes of integer/float
assert(stores_per_police_district.District.dtype == crime_socio.District.dtype)

#join in 
crime_socio_lq = pd.merge(left = crime_socio, right = stores_per_police_district, on = 'District', how ='left')
prevLen = len(crime_socio_lq)
crime_socio_lq = crime_socio_lq.dropna()  # KLUDGE check what the value is of dropped columns
endLen = len(crime_socio_lq)
cr_so_lq_bk = crime_socio_lq.copy() # backup data 

#done joining in on the liquor store data



_="""
Now we just store the variables we want and write to file  

"""


#the first row below is for the crime data
#the second row are the socio economic rows
#the third are the liquor store counts
result = crime_socio_lq[['hr', 'min', 'Latitude', 'Longitude', 'categoryCode', 'Community Area', 'District',
 'PERCENT AGED UNDER 18 OR OVER 64', 'PERCENT OF HOUSING CROWDED', 'PER CAPITA INCOME ',  'PERCENT AGED 16+ UNEMPLOYED', 'HARDSHIP INDEX', 'PERCENT HOUSEHOLDS BELOW POVERTY', 'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA',
 'LiquorStoreCount_District', 'month_', 'year_']].copy()

result['index_'] = result.index 

#write the data frame to file 
path = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data processed\crime_socio_lqr_'
pathLast = path.replace('\\', r'//')
writeDFToFile(dfs = [(result, 'data')], path_ = pathLast)  # this writes a .xlsx Excel file dfs is a list of tuples. Each tuple, the first value is the df, and the second value is the Excel sheet you put it in 






##adding in the the point
result_simplified = result[['index_', 'Latitude','Longitude']].copy()
result_simplified.rename(columns = {'Latitude':'latitude','Longitude':'longitude'}, inplace = True)
#values = getDF(loc = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime still need to explore\Map_of_Grocery_Stores_-_2013.csv') , sheetname = 'data') 
path4 = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime still need to explore\Map_of_Grocery_Stores_-_2013.csv')
values = pd.read_csv(path4, sep=',', engine='python')
values2 = values[['STORE NAME', 'LATITUDE', 'LONGITUDE']].copy()
values2.rename(columns = {'LATITUDE':'latitude', "LONGITUDE":'longitude', 'STORE NAME':'category'}, inplace = True)

df = result_simplified
values = values2 
nameToCall = "key_groceryStores"



# C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data processed
# C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data lookups
# do it on a completely different process 
#Things to consider: 
#1. split up the data frame like 10 times 
#2. loop through batches 
#3. only keep the index_, latitude, and longitude (so only 3 columns) 
#4. we can join in later. 
#Requirements:
#values is a dataframe and should have 3 columns [category, latitude, longitude]
#nameTocall -> should have no spaces and shouldn't be too long
#df -> must have an "index_" column, and a 'latitude' and 'longitude' column 
def KNN_(df, values, nameToCall): #values should have 3 columns [category, latitude, longitude]
    
    values['index_'] = values.index
    
    writeDFToFile(dfs=[( values[['index_','category']].copy()  , nameToCall )] , path_ = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data lookups\lookup_'+nameToCall+"_" )
    
    arr = values[['latitude', 'longitude']].copy().values.T
    
    #column to write 
    ctr = 0
    def functionToApply(row):
        point = np.array([[ row['latitude'] ],
                           [row['longitude']]])   
        dists = np.sum(((arr-point)**2), axis = 0)
        
        global ctr 
        ctr+=1
        if(ctr %10000 == 0):
            print(ctr)

        return np.argmax(dists.tolist())
    
    t1 = datetime.datetime.now()
    print("Starting at ", t1)


    df['closestc'] = df.apply(functionToApply, axis = 1)


    t2 = datetime.datetime.now()
    print("Ending at ", t2)



# ran the analysis on the grocery store information 
# have df , need to join in the result 
df = df[['index_', 'closestc']].copy()
df.rename(columns = {'closestc':'closestGroceryStore'}, inplace =True)
printNulls(df)
printNulls(result)
result_grocery = pd.merge(left = result, right = df, on = 'index_', how = 'left')

writeDFToFile(dfs=[( result_grocery , 'data' )] , path_ = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearningData\chicago energy usage\crime data processed\grocery_joined___'+nameToCall+"_" )




def test_KNN_():
    df = pd.DataFrame({'index_', ''})


_="""
Join in the nearest grocery store , do this in batches 

"""




#take the one that was written and then start from where you left off at 







_="""

"""

#Sunday
#1. JHU -> get the quiz done in morning/ get research assignment done/ final project brain storm
#2. Company -> put in 5 hours 
#3. SU -> 

#Brainstorm how to do CNN work # mapbox API to take a picture and send to the thing, 
#research if you can screen shot
#reserach if you can zoom in automatically 
#research if you can save file -> 
# filter on the image 
# take in the picture 


#For Tuesday: 




#Saturday 11/10/2018
#1. Amazon getting signed up.

#Sunday 11/11/2018  -> learn Tensorflow 
#Learn Tensorflow  -> pickling and non pickling as you go along

#join in another dataset after preprocessing everything 





#train the old network better
# join in more data -> so you can better your performance 
# see if you can improve performance 
# get tensorflow to connect to the GPU on this laptop 
#

# set up Amazon -> 


# sagemaker on AWS -> 

#realistically -> get the network to wo


# high level -> install tensorflow -
# take a snap shot of the imag
# take a screen shot of the image in JS -> and send the data 




#

#preprocessing the picture 