
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

####### Flask server

from flask import Flask, json, render_template, request , jsonify
#import ExcelImport as ei
import pandas as pd
#from Component import Component
from pandas import ExcelFile
from pandas import ExcelWriter
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pickle
#import json 
import pathlib 
import csv
import datetime 
import json as _json_

print("1")
data_path = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\cnn_data__2018-11-25 09-18-06.xlsx'
data_path = r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/cnn_data__testing_2018-11-25 11-07-00.xlsx'
data_path =cleanURL(data_path)
cnn_data  = getDF(loc = data_path, sheetname = 'cnn_data')

folderName = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\outputs\run1\cnn_data_'
folderName = cleanURL(folderName)

app = Flask(__name__)


#testing method
@app.route('/',methods = ['GET'])
def hello_world():
    global vals 
    lenVals = len(vals)
    lenVals = int(lenVals*.9)
    finalresult = json.dumps(vals)
    vals = vals[:lenVals]
    return finalresult


#sending the JSON file 
#steps: when requested, filter out the data frame, check if empty -> make an row with categoryCode = 34
#Todo: loadng static data for the map , KLUDGE 
@app.route('/dynamic_data',methods = ['GET']) # https://stackoverflow.com/questions/10434599/how-to-get-data-received-in-flask-request 
def dynamic_data_for_map():                     # https://stackoverflow.com/questions/13081532/return-json-response-from-flask-view
    #print(request.method)
    #print(request.args['asd'])
    #dataDict = request.get_json()
    #print(dataDict)
    #print(type(dataDict))
    #print(dataDict['okay'])
    #print(request.get_json())
    #print(request.args.get('asd'))

    #dataDict = request.get_json()
    #year = int(dataDict['year'])
    #week = int(dataDict['week'])
    #categoryCode = int(dataDict['categoryCode'])

    year = int(request.args.get('year'))
    week = int(request.args.get('week'))
    categoryCode =int(request.args.get('categoryCode'))

    global cnn_data
    df = cnn_data.loc[(cnn_data['year_'] ==year) & (cnn_data['week_'] ==week) & (cnn_data['categoryCode'] ==categoryCode)][['categoryCode', 'Latitude','Longitude']].copy()
    print("len of df is " , len(df))
    print("max lat: " , df['Latitude'].max())
    print("min lat: " , df['Latitude'].min())
    print("max long: " , df['Longitude'].max())
    print("min long: " , df['Longitude'].min())

    if(len(df) == 0):#for that year, week, categoryCode => we have nothing , just make 
        #make a nothing categoryCode
        pass 
    
    json__ = returnJson(df)

    #https://stackoverflow.com/questions/13081532/return-json-response-from-flask-view
    response = app.response_class(
        response=json__,
        status=200,
        mimetype='application/json'
    )
    
    print( json__) 
    return response

    #return json.dumps([1,2,3])


#Todo: loadng static data for the map , KLUDGE 
@app.route('/static_map_data',methods = ['GET'])
def static_data_for_map():
    #print(image_data)
    #print(len(image_data))
    print(image_data[-1])
    return json.dumps([1,2,3])


image_data=[["name", "base64"]]

@app.route('/storeImage',methods = ['POST'])
def store_base64Image():
    dataDict = request.get_json()
    image = dataDict['image']
    name = dataDict['name']
    

    global image_data
    image_data.append([name, image])

    if(len(image_data) == 300):
        #write to the file 
        write2dArrayToCSV()
        #empty out the image_data
        image_data=[["name", "base64"]]
        
    response = app.response_class(
        response="{\"success\":\"true\"}",
        status=200,
        mimetype='application/json'
    )
    return response
    #week = int(dataDict['week'])
    #categoryCode = int(dataDict['categoryCode'])





#loading the html file , maps.html 
@app.route('/<string:page_name>/')
def render_static(page_name):
    print(page_name)
    return render_template('%s.html' % page_name)


#sending data to the backend
# https://stackoverflow.com/questions/10434599/how-to-get-data-received-in-flask-request 
counter = 0
def write2dArrayToCSV():
    time_ = str(datetime.datetime.now())
    current_date_time = time_[0:time_.index(".")]
    current_date_time = current_date_time.replace(":", "-")
    global counter 
    
    fileNameLocation = folderName + current_date_time+"_"+str(counter)+".csv"
    with open(fileNameLocation,"w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(image_data)

    counter+=1

def returnJson(rowsDf):# going to have the categoryCode, Latitude, and Longitude
    #starting return on json
    print("starting return on json")
    rows = []
    arr = {}
    keysList =  rowsDf.irow(0).keys().tolist()
    for x in range(0, len(rowsDf)):
        arr = {}
        for key in keysList:
            arr[key] = rowsDf.irow(x)[key]
        rows.append(arr)
    print("ending return on json")
    if(len(rows) == 0):
        return json.dumps([{}])
    return json.dumps(rows)



app.run()
print("running")



# html loading works 
#below is the js request to be made  , next we need to be able to get in the parameters specified 
a = """
fetch('http://127.0.0.1:5000/', { 
  method: 'GET', 
  headers: {'Content-Type': 'application/json'}, 
  data: {'okay': 'then'}
})
.then(res => res.json())
.then(console.log)
"""

b = """
fetch('http://127.0.0.1:5000/dynamic_data', { 
  method: 'GET', 
  headers: {'Content-Type': 'application/json; charset=utf-8'}, 
  data: {"okay": "then"}
})
.then(res => res.json())
.then(console.log)
"""

c= """
fetch('http://127.0.0.1:5000/dynamic_data?asd=4', { 
  method: 'POST', 
  headers: {'Content-Type': 'application/json; charset=utf-8'}, 
  body: JSON.stringify({"okay": "then"})
})
.then(res => {console.log(res);res.json()})
.then(console.log)
"""

d = """
fetch('http://127.0.0.1:5000/dynamic_data?asd=4', { 
  method: 'POST', 
  headers: {'Content-Type': 'application/json; charset=utf-8'}, 
  body: JSON.stringify({"year": 2017, "week": 31, "categoryCode":32})
})
.then(res => {console.log(res.json());})
.then(console.log)
"""    

#year = int(request.args.get('year'))
#week = int(request.args.get('week'))
#categoryCode =int(request.args.get('categoryCode'))
e ="""
fetch('http://127.0.0.1:5000/dynamic_data?year=2017&week=31&categoryCode=32', { 
  method: 'GET', 
  headers: {'Content-Type': 'application/json; charset=utf-8'}, 
})
.then(res => {console.log(res.json()); window.wer = res;})
.then(console.log)
"""


f= """

fetch('http://127.0.0.1:5000/dynamic_data?year=2017&week=31&categoryCode=32', { 
  method: 'GET', 
  headers: {'Content-Type': 'application/json; charset=utf-8'}, 
})
.then(res => {return res.json()}).then(data => console.log(data))
"""

#for posting and image 
g = """
fetch('http://127.0.0.1:5000/storeImage', { 
  method: 'POST', 
  headers: {'Content-Type': 'application/json; charset=utf-8'}, 
  body: JSON.stringify({"name": "2017", "image": "23432342"})
})
.then(res => {return res.json()})
.then(data => console.log(data))
"""


#first pass 
# just get some simple images where we know the category and the dots -> 
# second pass 
# -> have about 10 for each data point -> and have bounding boxes for the cnn. like the yolov2 algorithm 

# about:debugging 
#http://127.0.0.1:5000/maps/
#https://support.mozilla.org/en-US/questions/905902


#http://www.pythoninformer.com/python-libraries/numpy/numpy-and-images/
#image processing: http://www.pythoninformer.com/python-libraries/numpy/numpy-and-images/ 
