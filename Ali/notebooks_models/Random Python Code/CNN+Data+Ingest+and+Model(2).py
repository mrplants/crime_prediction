
# coding: utf-8

# In[7]:


import tensorflow as tf
import zipfile
import os
import pandas as pd
import numpy as np
import pathlib


# In[8]:


def cleanURL(url):
    p = pathlib.Path(url)
    path = str(p.as_posix()) 
    return path 


def getDF(loc, sheetname):
    dataframe = pd.read_excel(loc, sheetname)
    #https://stackoverflow.com/questions/40950310/strip-trim-all-strings-of-a-dataframe
    dataframe = dataframe.applymap(lambda x: x.strip() if type(x) is str else x)
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
cwd = os.getcwd()
cwd


# In[9]:


# zipLoc = cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\drive-download-20181203T055115Z-005.zip')
# directory = cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\\')+'/'

# print(zipLoc)
# print(directory)


# In[10]:


# with zipfile.ZipFile(zipLoc, 'r') as zip_ref:
#     zip_ref.extractall(directory)


# In[11]:


# a = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\x__buildings_.npy'))
# #b = cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\y__c16_.npy')
# #a = np.load(b)
# a.shape


# CNN - based off of the paper

# In[12]:


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder(shape =[None, n_H0, n_W0, n_C0], dtype = np.float32, name="X")
    Y = tf.placeholder(shape  =[None, n_y], dtype = np.float32 , name="Y")
    ### END CODE HERE ###
    
    return X, Y


# In[13]:


X, Y = create_placeholders(64, 64, 3, 35)
print ("X = " + str(X))
print ("Y = " + str(Y))


# # First CNN Trial
# 
# ### First filter shape: (10 , 10,3 , 3) stride = 2, valid padding
#     
#     (64 +2p - f)/s +1 => (64+0-10)/2+1 = 28
#     So, (?, 64, 64, 3) * (10 , 10 , 3,  3) = (?, 28, 28, 3 )
#     
# ### Average Pooling Layer: (3 , 3 , 3) stride = 1, Padding = SAME
#     
#     (28 +2p - f)/s +1 => (28+2*1-3)/1+1 = 28
#     So, (?, 28, 28, 3) * ( 3 , 3,  3) = (?, 28, 28, 3 )
#     
#     
# ### Second filter shape: (6 , 6 ,3, 2) stride = 2, valid padding
#     
#     (28 +2p - f)/s +1 => (28+0-6)/2+1 = 12
#     So, (?, 28, 28, 3) * (6 , 6 , 3, 2) = (?, 12, 12, 2)
#     
# ### Max Pooling layer valid padding stride 1 (3,3)
#     
#     (12 +2p - f)/s +1 => (12+0-3)/1+1 = 10
#     So, (?, 12,12, 2) * (3, 3 , 2, 2) = (?, 10, 10, 2)
# 
#     
# ### Fourth filter shape: Flatten , fully connected (10*10*2) = 200
#     
#     W3 = 36 by 200
#     
# ### Softmax function for evaluation    
# 

# In[14]:


def initialize_parameters():    

    W1 = tf.get_variable("W1", [10, 10, 3, 3], initializer =tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [6, 6, 3, 2], initializer =tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [35,200] , initializer =tf.contrib.layers.xavier_initializer(seed = 0) )

    parameters = {"W1": W1, "W2": W2 , 'W3': 'W3'}
    
    return parameters


# In[15]:


def forward_prop(X, params):
    W1 = params['W1']
    W2 = params['W2']    
    W3 = params['W3']
    
    #convolution 
    Z1 = tf.nn.conv2d(X,W1, strides = [1,2,2,1], padding = 'VALID')
    
    #bias added automatically # RELU
    A1 = tf.nn.relu(Z1)
    
    #average pooling -> at this point all features/weights are important to us
    P1 = tf.nn.avg_pool(A1, ksize = [1,3,3,1], strides = [1,1,1,1], padding = 'SAME')

    # convolution 
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,2,2,1], padding = 'VALID')
    
    #RELU
    A2 = tf.nn.relu(Z2)
    
    #max pooling
    P2 = tf.nn.max_pool(A2, ksize = [1,3,3,1], strides = [1,1,1,1], padding = 'VALID')
    
    #flatten
    P2 = tf.contrib.layers.flatten(P2)

    #fully connected
    Z3 = tf.contrib.layers.fully_connected(P2, 35, activation_fn = None) #1 for yes/no
    #going to add the softmax directly

    return Z3


# # Run or Train the model

# In[ ]:


tf.reset_default_graph()

learning_rate = .008
costs = []
num_epochs = 20



# # Initialize all the variables globally
# init = tf.global_variables_initializer()

# Start the session to compute the tensorflow graph
with tf.Session() as sess:

    #the model
    X, Y = create_placeholders(64, 64, 3, 35)
    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)
    
    #init must be after optimizer
    init = tf.global_variables_initializer()

    
    # Run the initialization
    sess.run(init)

    # Do the training loop
    for epoch in range(num_epochs):

        minibatch_cost = 0.
        num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
        seed = seed + 1
        minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # IMPORTANT: The line that runs the graph on a minibatch.
            # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            ### START CODE HERE ### (1 line)
            _ , temp_cost = sess.run([optimizer,cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            ### END CODE HERE ###

            minibatch_cost += temp_cost / num_minibatches


# # Using the data loader methodology:

# In[322]:


################
# Constants 
################
BATCHSIZE_Y = 3000
LAST_BATCHSIZE_Y = 3000
TIMES_OF_DAY = 24
#The last day since the LEntries data only goes to 6/20/2018, we should filter to that end 
LAST_DAY = 6350 # in terms of timedelta.days
#FIRST_DAY was 1/1/2001, last day should then be 6/30/2018
MINIBATCHES_AMT = 762 #or (200,762) (254 batches of 600) (gcf of 3000 and 152400) # keep in mind that this number is going to be scaled by 16 since 16 64by64 images are in one 256by256
#150 or 1016              #batch size should never be more than 2900 (really 3000, but to stay on the safe side)
iHeight = 256
iWidth = 256


################
# Load all the data , for the larger data values , just run  
################

# load all the data
datesb = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\dates_data_b.npy'))
dates = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\dates_data.npy'))
buildings = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__buildings_b.npy'))

#make sure there are no nan values in buildings 
mask = np.isnan(buildings)
indices = np.where(mask ==True)
z = indices[0]
y = indices[1]
x = indices[2]
buildings[z,y,x] = -1

businesses = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__businesses_b.npy'))
socio = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__socio_b.npy'))
lentries = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__Lentries_c.npy'), mmap_mode  = 'r')
waterway = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\waterway.npy'))

#get all the masks used

################
# Preprocess every image value to be its transpose
################
# buildings  buildingsT -> buildingsTStacked
# businesses    businessesT -> businessesTStacked
# socio   socioT -> socioTStacked
buildingsT= transpose3dImage(buildings)
businessesT= transpose3dImage(businesses)
socioT = transpose3dImage(socio)

#all the outputs
outputsData =[]
for i_ in range(1, 53):
    y_ = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\y__c'+str(i_)+'_c.npy'), mmap_mode  = 'r')
    outputsData.append(y_)


# In[ ]:


# to print the waterway mask
import matplotlib.pyplot as plt
import numpy as np

img =waterway
arr = []
for a in img:
    arr = [a] + arr
plt.pcolor( arr, cmap = 'gist_ncar' )

plt.show()


# ### Cleaning up the data (to align all the indices of the data)

# In[326]:


lentries.shape
print(len(lentries)*24)
print(len(lentries))
# len of everything 
print('number of outputs ' , BATCHSIZE_Y*51+len(outputsData[-1])) # number of outputs

# calculate the exact value of how many (256 by 256 by numlayer) images we should have 
indices = LAST_DAY* TIMES_OF_DAY
indices_ = np.array([index for index in range(indices)])
print('Actual data points', len(indices_)) # 0 - 153,335 or 153,336 values                      #152400
print("will shave off extra data. Everything starts from the same point in time")


# ### Generate minibatches. Shuffle the minibatches, have a function to return a minibatch of X and Y

# In[327]:


print('Indices of datapoints', indices_)
#gets the indices of the minibatches
minibatches = np.split(indices_,MINIBATCHES_AMT)
sample = np.array(minibatches[0])
print('sample minibatch: ' , sample )

#to shuffle the minibatches for random minibatches  we will do this every epoch
np.random.shuffle(minibatches)



# In[328]:


# datesb -- no need 
# dates -- no need
# buildings  buildingsT -> buildingsTStacked
# businesses    businessesT -> businessesTStacked
# socio   socioT -> socioTStacked
# lentries 
# waterway  -- not part of data 
# outputsData 


#next steps -> add lentries, temperature, and masks 

# now we write a function that will return to us the correct minibatch , with all the image data generated
def generateMinibatch(minibatchIndices):    #everything must be transposed
    #general steps:
    #get the x inputs
    #    same as the file of text 
    
    #step 1. dates (make 12 layers of month, day, year , timeOfDay)  # dateLayers don't need to be transposed -> just 1 value
    dateLayers = generateDatesLayers(minibatchIndices, datesb) # (150, 256, 256, 4)  => len(minibatchIndices) = 150
    
    #step 2. Buildings 10 layers 
    buildingLayers = stackManyTimes(buildingsT, len(minibatchIndices))  # buildingsT should be (256,256,10) and result should (150, 256, 256, 10)
  
    #step 3. 
    businessesLayers = stackManyTimes(businessesT, len(minibatchIndices))
    
    #step 4. L entries
    #come back to this one   => must be transposed!!!
    LentryLayers = None
    
    #step 5. socio
    socioLayers = stackManyTimes(socioT, len(minibatchIndices))
    
    #step 6. temperature #should be format of
    #pass on this for now -> will add this for later iterations
    
    #step 7. concat everything
    inputImage = np.concatenate((dateLayers,buildingLayers,businessesLayers,socioLayers) , axis = -1)
    print(np.where((inputImage[:4]==dateLayers)==False))
    print(np.where((inputImage[4:14]==buildingLayers)==False))

    #step 7. outputs
    output_image = calculateOutput(sample)
    return inputImage, output_image


# In[329]:


inputImage, output_image = generateMinibatch(minibatches[10])


# In[334]:


print(inputImage.shape)
print(output_image.shape)

imagesList = np.split(inputImage, 4, axis = 1)
imagesList[0].shape


# In[335]:


images64by64 = []
for almostImage in imagesList:
    imagesList64by64 = np.split(almostImage, 4, axis = 2) # axis is 0, 1, 2
    for actual64by64 in imagesList64by64:
        images64by64.append(actual64by64)


# In[337]:


print(len(images64by64))
images64by64[0].shape


# In[338]:


for i,_64 in enumerate(images64by64):
    print(i, _64.shape)


# In[332]:



imagesList = np.split(inputImage, 4, axis = 1)
images64by64 = []

for almostImage in imagesList:
    imagesList64by64 = np.split(almostImage, 4, axis = 2) # axis is 0, 1, 2
    for actual64by64 in imagesList64by64:
        images64by64.append(actual64by64)

for i,_64 in enumerate(images64by64):
    print(i, _64.shape)





# Testing Dates

# In[154]:


#exploring things
base_img_mask = np.ones((2, 2), dtype=np.float32)
print(base_img_mask.shape)
base_img_mask[0,0] = 1
base_img_mask[0,1] = 2
base_img_mask[1,0] = 3
base_img_mask[1,1] = 4

base_img_mask

np.array([base_img_mask]*4)


# In[214]:


###Testing Dates 

base_img_mask = np.ones((256, 256), dtype=np.float32)
xy = np.dstack([base_img_mask]*4)
print(xy.shape)

dateLayers = datesb[minibatchIndices]
print(dateLayers.shape)
dateLayersReshaped = dateLayers.reshape((len(dateLayers), 1,1,dateLayers.shape[1]))
print(dateLayersReshaped.shape)


xyz = xy* dateLayersReshaped 
xyz.shape

#do some checks 
firstVal = dateLayers[45][1] #last layer, 2nd d value
print(firstVal)

print(set(xyz[45, :,:, 1].flatten()))
#change the 45 and the 1 and see if things match! 

def generateDatesLayers(minibatchIndices, datesb):
    base_img_mask = np.ones((256, 256), dtype=np.float32)
    xy = np.dstack([base_img_mask]*4) # shape (256, 256, 4)
    dateLayers = datesb[minibatchIndices] # (len(minibatchIndices) , 4)
    dateLayersReshaped = dateLayers.reshape((len(dateLayers), 1,1,dateLayers.shape[1])) # (len(minibatchIndices),1,1 , 4)
    xyz = xy* dateLayersReshaped 
    return xyz

    


# Testing Copy Generation of an image ( numLayers, 256, 256) to  (256, 256, numLayers) and then to  (sampleNum, 256, 256, numLayers) 

# In[229]:


# Removing Null values
mask = np.isnan(buildings)
indices = np.where(mask ==True)
z = indices[0]
y = indices[1]
x = indices[2]
buildings[z,y,x] = -1

# mask2 = np.isnan(buildingsTest)
# indices2 = np.where(mask2 ==True)
# print(indices2)
# print(len(indices))
# print((indices[0].flatten().tolist()))


# In[271]:


#For transposing images 

testData = np.array([[[1,2],[3,4]],
                    [[5,6],[7,8]],
                    [[9,10],[11,12]]])
print(testData)
print(testData.shape)

def transpose3dImage(img):
    img_T = img.T #tested this actually does what we wante it to do.  
    return img_T

testDataT = transpose3dImage(testData) # ( numLayers, 256, 256) to (256, 256, numLayers) 

print(testDataT[:,:,1])
print(testDataT.shape)
# dateLayers = generateDatesLayers(minibatchIndices, datesb)
# dateLayers.shape


# In[270]:


# For  (256, 256, numLayers) and then to (sampleNum, 256, 256, numLayers) 

def stackManyTimes(_3dimg,times):
    _3dimg_shape = _3dimg.shape
    result = np.zeros(shape=(times, _3dimg_shape[0], _3dimg_shape[1],_3dimg_shape[2] ))
    for x in range(times):
        result[x] = _3dimg
    return result
    
buildings2 = np.array(transpose3dImage(buildings))
print(buildings2.shape)

buildings3_multiplied = stackManyTimes(buildings2, 4)
print(buildings3_multiplied.shape)

np.where((buildings3_multiplied[3]==buildings2) == False)


# ### Getting the right output from the list of 52, given that we know that batch size is 

# In[315]:


# Getting the right output from the list of 52, we know that batch size is 
# print(minibatchIndices[])
# dateLayers[-1][2]
print(len(outputsData))

outputsData[51].shape

sample = minibatches[0]
batchMin = min(sample)#67050
batchMax = max(sample)#67199
batchMax  = 69050
print("batch min " , batchMin," batch max ", batchMax )
remMin = batchMin % BATCHSIZE_Y
multipleMin = int(batchMin / BATCHSIZE_Y)

remMax = batchMax % BATCHSIZE_Y
multipleMax = int(batchMax / BATCHSIZE_Y)


print(multipleMin," ",multipleMax)
# outputsData[multipleMin-1] # 1 is really 0, 2 is really 1 , and so on. 


# batchMin
#152300 - 152399
# 155900 to 155917
int(155900/BATCHSIZE_Y)
print(remMax)


# In[316]:


batch = None
if(multipleMin != multipleMax):# have to concatenate  batch size can never be more than 3000#
    print('here')
    batch = np.concatenate((outputsData[multipleMin],outputsData[multipleMax]), axis = 0)
else:
    print('not here')
    batch = outputsData[multipleMin]
    
offset = multipleMin*BATCHSIZE_Y#+remMin

print(offset)
batch.shape
sample_ = np.array(sample)-offset

data_output = batch[sample_]


# In[310]:


data_output.shape

#very sure that we have to transpose everything -> since we transposed everything in the input layer 
for i_, img in enumerate(data_output):
    data_output[i_] = img.T

#now we make the 16, by 16 images 

sample = minibatches[0]

def calculateOutput(sample):
    batchMin = min(sample)#67050
    batchMax = max(sample)#67199
    #     batchMax  = 69050
    #print("batch min " , batchMin," batch max ", batchMax )
    remMin = batchMin % BATCHSIZE_Y
    multipleMin = int(batchMin / BATCHSIZE_Y)

    remMax = batchMax % BATCHSIZE_Y
    multipleMax = int(batchMax / BATCHSIZE_Y)
    
    batch = None
    if(multipleMin != multipleMax):# have to concatenate  batch size can never be more than 3000#
        print('here')
        batch = np.concatenate((outputsData[multipleMin],outputsData[multipleMax]), axis = 0)
    else:
        print('not here')
        batch = outputsData[multipleMin]

    offset = multipleMin*BATCHSIZE_Y
    sample_ = np.array(sample)-offset
    data_output = batch[sample_]
    return data_output


output_images = calculateOutput(sample)
output_images.shape
#( numLayers, 256, 256) to (256, 256, numLayers) 
#testDataT = transpose3dImage(testData)


# In[314]:


tf.reset_default_graph()

learning_rate = .008
costs = []
with tf.Session() as sess:
    
    X, Y = create_placeholders(64, 64, 3, 35)
    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)
    #init must be after optimizer
    init = tf.global_variables_initializer()

    sess.run(init)
    
    for epoch in range(10):
        _ , temp_cost = sess.run([optimizer,cost], feed_dict={X: np.random.randn(2,64,64,3), Y: np.random.randn(2,35)})
        costs.append(temp_cost)
        
        
    #     a = sess.run(cost, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,35)})
    #     print("Z3 = " + str(a))
costs

