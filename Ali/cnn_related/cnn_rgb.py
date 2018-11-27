#https://pythonspot.com/matplotlib-save-figure-to-image-file/
#http://www.pythoninformer.com/python-libraries/numpy/numpy-and-images/
#https://stackoverflow.com/questions/15160123/adding-a-background-image-to-a-plot-with-known-corner-coordinates


import os 
import sys
import copy
import pandas as pd
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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


# Get the CNN data 
#############################################################
data_path = r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/cnn_data__11_25_2018.csv'
#data_path = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\cnn_data__2018-11-25 09-18-06.xlsx'
#data_path = r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/cnn_data__testing_2018-11-25 11-07-00.xlsx'
data_path =cleanURL(data_path)
#cnn_data  = getDF(loc = data_path, sheetname = 'cnn_data')
cnn_data = pd.read_csv(data_path, sep=',', engine='python')

folderName = r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\outputs\run1\cnn_data_'
folderName = cleanURL(folderName)



datafile = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\base_image1_no_labels.png')
#datafile2= cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\images_gen\image1.png')
#img = imread(datafile)
image = Image.open(datafile)
imagebk  =copy.deepcopy(image)

nparr = np.array(image)
#draw = ImageDraw.Draw(image)

#xy = [(10, 10), (20,20)] #, (10, 20), (20,10)]
#outline = '#FF0000'
#fill = '#FF0000'
#draw.ellipse(xy, outline, fill)

XMIN_long = -87.906463155
XMAX_long = -87.526977196
YMIN_lat = 41.653021074
YMAX_lat = 42.019883291

XMIN_pixel = 0
XMAX_pixel = nparr.shape[0] - 1
YMIN_pixel = 0
YMAX_pixel = nparr.shape[1] - 1

pixels_to_degrees_x =  (XMAX_pixel - XMIN_pixel)/(XMAX_long -XMIN_long)
pixels_to_degrees_y =  (YMAX_pixel - YMIN_pixel)/(YMAX_lat -YMIN_lat)

dotSize = 6
dotOffset = dotSize/2


categoryCodeColors = {0:'#fbb03b',
1:'#3bb2d0',
2:'#bcaaa4',
3:'#5e35b1',
4:'#7b1fa2',
5:'#CCFF33',
6:'#FF0099',
7:'#33FFCC',
8:'#66FFFF',
9:'#990066',
10:'#996699',
11:'#1B2631',
12:'#D4AC0D',
13:'#7D6608',
14:'#76D7C4',
15:'#9B59B6',
16:'#5d6d7e',
17:'#5f6a6a',
18:'#f5b7b1',
19:'#aed6f1',
20:'#edbb99',
21:'#d35400',
22:'#17a589',
23:'#2471a3',
24:'#943126',
25:'#aeb6bf',
26:'#283747',
27:'#d35400',
28:'#FFC300',
29:'#DAF7A6',
30:'#7fb3d5',
31:'#2471a3',
32:'#2874a6',
33:'#f1c40f'}




xy = None
outline = None 
fill = None 

def calculateXY(row):
    lat = row['Latitude']
    long = row['Longitude']
    catCode = row['categoryCode']
    color = categoryCodeColors[catCode]
    x_val = int((long - XMIN_long)*pixels_to_degrees_x)
    y_val = int(  YMAX_pixel -  (lat - YMIN_lat)*pixels_to_degrees_y)
    xy = [(x_val-dotOffset, y_val-dotOffset), (x_val+dotOffset, y_val+dotOffset)]
    draw.ellipse(xy, outline, fill)



base_dir = cleanURL(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/images_gen2/im_')

for year in range(2001, 2019):
    for week in range(1, 54):
        for categoryCode in range(0,34): # 0 - 33 
            image = copy.deepcopy(imagebk)
            df = cnn_data.loc[(cnn_data['year_'] ==year) & (cnn_data['week_'] ==week) & (cnn_data['categoryCode'] ==categoryCode)][['categoryCode', 'Latitude','Longitude']].copy() 
            df = df.head(200) #for less data for faster running 
            draw = ImageDraw.Draw(image)
            outline = categoryCodeColors[categoryCode]
            fill = categoryCodeColors[categoryCode]
            df.apply(calculateXY, axis = 1)
            fileName= base_dir + str(year) + "_"+str(week) + "_" + str(categoryCode)+"_.png"
            image.save(fileName)
            



#######################################################
#saving everything
#don't need to have the \ after the foldername 
def searchDir(path = r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/images_gen2/'):
    rootdir = cleanURL(path)
    filepaths= []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepaths.append(rootdir + "/" + file)
    return filepaths




filepaths_ = searchDir()

filepaths_1 = filepaths_[:10000]
filepaths_2 = filepaths_[10000:20000]
filepaths_3 = filepaths_[20000:30000]
filepaths_4 = filepaths_[30000:]

hugeArrX = []
hugeArrY = []
count = 0
for imagePath in filepaths_1:
    imageData = Image.open(imagePath)
    hugeArrX.append(np.array(imageData))
    yCategory = int(imagePath[imagePath.rindex('im'):].split('_')[-2])
    hugeArrY.append(yCategory)
    count+=1
    if(count %1000 ==0):
        print(count)

hugeArrX = np.array(hugeArrX)
hugeArrY = np.array(hugeArrY)
hugeArrY = hugeArrY.reshape((len(hugeArrY), 1))

np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2A_x.npy', hugeArrX)
np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2A_y.npy', hugeArrY)

################################

hugeArrX = None 
hugeArrY = None
hugeArrX = []
hugeArrY = []
for imagePath in filepaths_2:
    imageData = Image.open(imagePath)
    hugeArrX.append(np.array(imageData))
    yCategory = int(imagePath[imagePath.rindex('im'):].split('_')[-2])
    hugeArrY.append(yCategory)
    count+=1
    if(count %1000 ==0):
        print(count)

hugeArrX = np.array(hugeArrX)
hugeArrY = np.array(hugeArrY)
hugeArrY = hugeArrY.reshape((len(hugeArrY), 1))

np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2B_x.npy', hugeArrX)
np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2B_y.npy', hugeArrY)


#################################

hugeArrX = None 
hugeArrY = None
hugeArrX = []
hugeArrY = []
for imagePath in filepaths_3:
    imageData = Image.open(imagePath)
    hugeArrX.append(np.array(imageData))
    yCategory = int(imagePath[imagePath.rindex('im'):].split('_')[-2])
    hugeArrY.append(yCategory)
    count+=1
    if(count %1000 ==0):
        print(count)

hugeArrX = np.array(hugeArrX)
hugeArrY = np.array(hugeArrY)
hugeArrY = hugeArrY.reshape((len(hugeArrY), 1))

np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2C_x.npy', hugeArrX)
np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2C_y.npy', hugeArrY)

####################################

hugeArrX = None 
hugeArrY = None
hugeArrX = []
hugeArrY = []
for imagePath in filepaths_4:
    imageData = Image.open(imagePath)
    hugeArrX.append(np.array(imageData))
    yCategory = int(imagePath[imagePath.rindex('im'):].split('_')[-2])
    hugeArrY.append(yCategory)
    count+=1
    if(count %1000 ==0):
        print(count)

hugeArrX = np.array(hugeArrX)
hugeArrY = np.array(hugeArrY)
hugeArrY = hugeArrY.reshape((len(hugeArrY), 1))

np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2D_x.npy', hugeArrX)
np.save(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2D_y.npy', hugeArrY)

######################################






datasetx = np.load(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2_x.npy')
datasety = np.load(r'C:/Users/j70514/Documents/Data Science Stuff/DeepLearning_cs230/cnn_data/CNN_DATASET_2_y.npy')

#https://stackoverflow.com/questions/43486077/how-to-get-image-from-imagedraw-in-pil 






#import matplotlib
#import matplotlib.pyplot as plt
#import numpy as np
 
#y = [2,4,6,8,10,12,14,16,18,20]
#x = np.arange(10)
#fig = plt.figure()
#ax = plt.subplot(111)
#ax.plot(x, y, label='$y = numbers')
#plt.title('Legend inside')
#ax.legend()
##plt.show()
 
#fig.savefig('plot.png')


##add a background to matplotlib plot 

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.misc import imread
#import matplotlib.cbook as cbook

#np.random.seed(0)
#x = np.random.uniform(0.0,10.0,15)
#y = np.random.uniform(0.0,10.0,15)

#datafile = cbook.get_sample_data('lena.jpg')
#img = imread(datafile)
#plt.scatter(x,y,zorder=1)
#plt.imshow(img, zorder=0, extent=[0.5, 8.0, 1.0, 7.0])
#plt.show()



#C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\base_image1_no_labels.png



#############################################################

#get the background image 
datafile = cleanURL(r'C:\Users\j70514\Documents\Data Science Stuff\DeepLearning_cs230\cnn_data\base_image1_no_labels.png')
img = imread(datafile)



x = np.random.uniform(0.0,img.shape[1],15)
y = np.random.uniform(0.0,img.shape[0],15)


fig = plt.figure()
plt.scatter(x,y,zorder=1)
plt.imshow(img, zorder=0, aspect='equal')
#plt.axis('off')
fig.subplots_adjust(bottom = 0)
fig.subplots_adjust(top = 1)
fig.subplots_adjust(right = 1)
fig.subplots_adjust(left = 0)
#plt.show()

fig.savefig(folderName+'.png',  bbox_inches = "tight")
#fig.savefig(folderName+'.png' , bbox_inches = "tight", pad_inches = 0)


##Alternative approach

fig = plt.figure()
plt.scatter(x,y,zorder=1)
plt.imshow(img, zorder=0, aspect='equal')
#plt.axis('off')
#fig.subplots_adjust(bottom = 0)
#fig.subplots_adjust(top = 1)
#fig.subplots_adjust(right = 1)
#fig.subplots_adjust(left = 0)
#plt.show()

#fig.savefig(folderName+'.png',  bbox_inches = "tight")
fig.savefig(folderName+'.png', bbox_inches='tight',transparent=True, pad_inches=0)


#Alternative Approach (Second Best one)

fig = plt.figure()
plt.scatter(x,y,zorder=1)
plt.imshow(img, zorder=0, aspect='equal')
plt.axis('off')
fig.subplots_adjust(bottom = 0)
fig.subplots_adjust(top = 1)
fig.subplots_adjust(right = 1)
fig.subplots_adjust(left = 0)
#plt.show()

#fig.savefig(folderName+'.png',  bbox_inches = "tight")
fig.savefig(folderName+'.png', bbox_inches='tight',transparent=True, pad_inches=0)



#Alternative Approach 


fig = plt.figure(frameon = False)

ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.scatter(x,y,zorder=1)
ax.imshow(img, zorder=0, aspect='equal')
fig.savefig(folderName+'.png')
#plt.show
#fig.savefig(folderName+'.png',  bbox_inches = "tight")
#fig.savefig(folderName+'.png', bbox_inches='tight',transparent=True, pad_inches=0)



#############################################################