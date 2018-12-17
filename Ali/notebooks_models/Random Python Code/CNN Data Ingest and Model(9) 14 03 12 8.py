
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import tensorflow as tf
import zipfile
import os
import pandas as pd
import numpy as np
import pathlib
import datetime
from collections import Counter
import copy
import matplotlib.pyplot as plt


# # Helpful Functions

# In[10]:


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


# # Load Data

# ### Using the data loader methodology:

# In[7]:


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

# load all the data #CHANGEME
# datesb = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\dates_data_b.npy'))
# dates = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\dates_data.npy'))
# buildings = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__buildings_b.npy'))
datesb = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/dates_data_b.npy'))
dates = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/dates_data.npy'))
buildings = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/x__buildings_b.npy'))



#make sure there are no nan values in buildings 
mask = np.isnan(buildings)
indices = np.where(mask ==True)
z = indices[0]
y = indices[1]
x = indices[2]
buildings[z,y,x] = -1

# businesses = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__businesses_b.npy'))
# socio = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__socio_b.npy'))
# lentries = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\x__Lentries_c.npy'), mmap_mode  = 'r')
# waterway = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\waterway.npy'))
businesses = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/x__businesses_b.npy'))
socio = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/x__socio_b.npy'))
lentries = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/x__Lentries_c.npy'))#, mmap_mode  = 'r')#CHANGEME
waterway = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/waterway.npy'))
##KLUDGE: This is a quick fix, just set a lot of the data to 0
# waterway[:,170:]= 1

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

print('done0')


# In[11]:


waterway = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/waterway.npy'))
plt.imshow(waterway)
plt.show()


# In[12]:


#all the outputs
outputsData =[]
for i_ in range(1, 53): # /home/muhammadayub/Desktop/CS230/training_data/outputs/y__c2_127.npy
    y_ = np.load(cleanURL(r'/home/muhammadayub/Desktop/CS230/training_data/outputs/y__c'+str(i_)+'_127.npy'), mmap_mode  = 'r')
    outputsData.append(y_)


# In[13]:


# if at all possible  #CHANGEME
outputsDataReal = np.concatenate(outputsData, axis = 0)
print(outputsDataReal.shape)
for i_ in range(len(outputsData)):
    outputsData[i_] =None
    
# if at all possible  #CHANGEME
for i_ in range(len(outputsDataReal)):
    outputsDataReal[i_] =outputsDataReal[i_].T
print('run once')
# print(outputsDataReal.max())#33 is max, 0 is min
# print(outputsDataReal.min())


# In[37]:


# to print the waterway mask
import matplotlib.pyplot as plt
import numpy as np

def plotImg(img):
    if(type(img) ==type(None)):
        img =outputsDataReal[10000]
    arr = []
    for a in img:
        arr = [a] + arr
    plt.pcolor( arr, cmap = 'gist_ncar' )

    plt.show()


# In[39]:


abc = np.where(outputsDataReal[1000:10000] !=0)
# plotImg(outputsDataReal[1000])
len(abc[0])


# ### Cleaning up the data (to align all the indices of the data)

# In[ ]:


lentries.shape
print(len(lentries)*24)
print(len(lentries))
# len of everything 
print('number of outputs ' , BATCHSIZE_Y*51+len(outputsData[-1])) # number of outputs


# In[14]:



# calculate the exact value of how many (256 by 256 by numlayer) images we should have 
indices = LAST_DAY* TIMES_OF_DAY
indices_ = np.array([index for index in range(indices)])
print('Actual data points', len(indices_)) # 0 - 153,335 or 153,336 values                      #152400
print("will shave off extra data. Everything starts from the same point in time")


# #### Generate minibatches. Shuffle the minibatches, have a function to return a minibatch of X and Y

# In[15]:


print(MINIBATCHES_AMT)
train_split = int(MINIBATCHES_AMT*.90)
dev_split = int(MINIBATCHES_AMT*.05)
test_split = MINIBATCHES_AMT - train_split - dev_split

print(train_split, dev_split, test_split)
assert(train_split + dev_split + test_split)


# In[16]:


print('Indices of datapoints', indices_)
#gets the indices of the minibatches

minibatches = np.split(indices_,MINIBATCHES_AMT)
backup_minibatches = copy.deepcopy(minibatches)

devMiniBatches = minibatches[train_split:train_split+dev_split]
testMiniBatches = minibatches[train_split+dev_split:-1]# remove the last minibatch
minibatches = minibatches[:train_split] #training set  #must be at the end

# sample = np.array(minibatches[0])
# print('sample minibatch: ' , sample )
# #to shuffle the minibatches for random minibatches  we will do this every epoch
# np.random.shuffle(minibatches)
# print(devMiniBatches)
# print(testMiniBatches)


# In[275]:


len(devMiniBatches+testMiniBatches+minibatches)
inputImage = None
output_image = None


# In[17]:


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
#     print(np.where((inputImage[:4]==dateLayers)==False))
#     print(np.where((inputImage[4:14]==buildingLayers)==False))

    #step 7. outputs
    output_image = outputsDataReal[minibatchIndices]  # calculateOutput(minibatchIndices)
    return inputImage, output_image#, waterway

def generateDatesLayers(minibatchIndices, datesb):
    base_img_mask = np.ones((256, 256), dtype=np.float32)
    xy = np.dstack([base_img_mask]*4) # shape (256, 256, 4)
    dateLayers = datesb[minibatchIndices] # (len(minibatchIndices) , 4)
    dateLayersReshaped = dateLayers.reshape((len(dateLayers), 1,1,dateLayers.shape[1])) # (len(minibatchIndices),1,1 , 4)
    xyz = xy* dateLayersReshaped 
    return xyz

def transpose3dImage(img):
    img_T = img.T #tested this actually does what we wante it to do.  
    return img_T


def stackManyTimes(_3dimg,times):
    _3dimg_shape = _3dimg.shape
    result = np.zeros(shape=(times, _3dimg_shape[0], _3dimg_shape[1],_3dimg_shape[2] ), dtype=np.float32)
    for x in range(times):
        result[x] = _3dimg
    return result

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

def splitWaterWay(waterwayImg):
    imagesList = np.split(waterwayImg, 4, axis = 0)
    images64by64 = []

    for almostImage in imagesList:
        imagesList64by64 = np.split(almostImage, 4, axis = 1) # axis is 0, 1, 2
        for actual64by64 in imagesList64by64:
            images64by64.append(actual64by64)
    #     for i,_64 in enumerate(images64by64):
    #         print(i, _64.shape)
    return np.concatenate(images64by64, axis = 0)

def split256by256StackOnAxis(inputImg):  #returns the images in (64 , 64, numberOfChannels)
    imagesList = np.split(inputImg, 4, axis = 1)
    images64by64 = []

    for almostImage in imagesList:
        imagesList64by64 = np.split(almostImage, 4, axis = 2) # axis is 0, 1, 2
        for actual64by64 in imagesList64by64:
            images64by64.append(actual64by64)
    #     for i,_64 in enumerate(images64by64):
    #         print(i, _64.shape)
    return np.concatenate(images64by64, axis = 0)

def getOutputYVector(y_output_image64, numCats): # y_output_image64 is of shape (3200,64,64)  , 34 categories + -1
    result = np.zeros((len(y_output_image64), numCats), dtype=np.float32)
    cCount = None 
    for i_, img in enumerate(y_output_image64):
        cCount = Counter(img.flatten())
        if(len(np.unique(img)) == 1): #if all -1's
            result[i_][0] = 1
        else: # get the second most common thing (since -1 are going to be the most common)
            result[i_][int(cCount.most_common()[1][0])+1] = 1 # get the second value (index of 1) of most common array
    return result

    #get the argmax for now  -1 goes to 0, 0 goes to 1, etc. until you have 33 going to 34   -> kludge need to set this to -1
    #result[i_][int(cCount.most_common()[0][0])+1] = 1

def transformTo64(inputImage, output_image, numCats):
    inputImage64 =  split256by256StackOnAxis(inputImage)
    output_image64 = split256by256StackOnAxis(output_image)
    output_image64 = getOutputYVector(output_image64, numCats) #34 categories since we use -1s, but then shave them off
    return inputImage64, output_image64

def generateWaterWayMask(waterway, threshold):# between 0 and 4096, need atmost 200 to be water 
    sizeOfOneMiniBatch = int(len(indices_)/MINIBATCHES_AMT)
    waterwayMask = np.zeros((sizeOfOneMiniBatch, 256,256), dtype=np.float32)
    for i_ in range(len(waterwayMask)):
        waterwayMask[i_] = waterway
        
    #now that we have the ( 200, 256,256)  (sizeOfOneMiniBatch is 200 for example)
    #we can break it up into the 64 by 64 images
    waterwayMask64 = split256by256StackOnAxis(waterwayMask)
    #from here, you count up each one of the 64 by 64 images and set to the threshold
    waterwayMaskIndices = [] # include if they are one 
    for i_ in range(len(waterwayMask64)):
        if(np.sum(waterwayMask64[i_]) <= threshold): #> would mean you have more 1's than allowed -> not tolerable 
            waterwayMaskIndices.append(i_)
    return waterwayMaskIndices


# # Model Definition (based off research paper)

# In[18]:


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


# In[19]:


X, Y = create_placeholders(64, 64, 26, 35)  #will have 35 , 34 real values and 1 fake value
print ("X = " + str(X))
print ("Y = " + str(Y))


# ## First CNN Trial
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

# In[20]:


def initialize_parameters():    

    W1 = tf.get_variable("W1", [10, 10, 26, 3], initializer =tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [6, 6, 3, 2], initializer =tf.contrib.layers.xavier_initializer(seed = 0))
    W3 = tf.get_variable("W3", [35,200] , initializer =tf.contrib.layers.xavier_initializer(seed = 0) )

    parameters = {"W1": W1, "W2": W2 , 'W3': 'W3'}
    
    return parameters


# In[26]:


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

# In[27]:


# tf.reset_default_graph()
base_filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model2"
filepath =None # put in global namespace for ease of using later
learning_rate = .008
costs = []
num_epochs = 50

WaterWayMask = generateWaterWayMask(waterway, 400)

graph = tf.Graph()
with graph.as_default(): # https://stackoverflow.com/questions/36281129/no-variable-to-save-error-in-tensorflow

    #variable declarations must be before tf.train.Saver() unless here: https://stackoverflow.com/questions/50974976/tensorflow-why-must-saver-tf-train-saver-be-declared-after-variables-are
    #the model
    X, Y = create_placeholders(64, 64, 26, 35)
    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)

    
    saver = tf.train.Saver()

    # # Initialize all the variables globally
    # init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:


        #init must be after optimizer
        init = tf.global_variables_initializer()

        # Run the initialization
        sess.run(init)
        print('Starting ' , datetime.datetime.now())
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            np.random.shuffle(minibatches) # get new minibatch results 

            for i_ , minibatch in enumerate(minibatches[:80]):

                # Select a minibatch
                inputImage, output_image = generateMinibatch(minibatch)
                inputImage64, output_image64 = transformTo64(inputImage, output_image,35)
                output_image64 = output_image64[WaterWayMask]
                inputImage64 = inputImage64[WaterWayMask]


                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer,cost], feed_dict={X: inputImage64, Y: output_image64})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / MINIBATCHES_AMT

                print('Minibatch End: ', datetime.datetime.now())
                print(temp_cost,' ' , epoch,' ' ,i_)
                
                if(i_ % 20 == 0 ):
                    #save model
                    filepath = base_filepath+"/model_12_7__"+str(epoch)+"_"+str(i_)+".ckpt"
                    print(temp_cost, " at file name: ", filepath[25:])
                    save_path = saver.save(sess, filepath)

            costs.append(minibatch_cost)
            print('Epoch End: ', datetime.datetime.now())


# if((epoch ==0) and (i_==0)):
#     globalA = inputImage64
#     globalB = output_image64


# # Storing the Costs, etc. 

# In[148]:


costsNPArray = np.array(costs)
epochXaxis = np.arange(len(costs))
plt.plot(epochXaxis, costsNPArray)
plt.show()

costsFilePath = '/home/muhammadayub/Desktop/CS230/models_saved/model2/model_12_7__epochsNum_'+str(num_epochs)+'.npy'
np.save(costsFilePath, costsNPArray )


# In[149]:


np.load(costsFilePath)
# filepath


# # This is to figure out the F1 multi class label predictions 

# In[73]:


nb = 5
dims =3

labels = (np.random.randn(nb, dims) > 0.5).astype(int)
predictions = (np.random.randn(nb, dims) > 0.5).astype(int)
y_true = labels
y_pred = predictions
print(labels)
print(predictions)




# In[74]:


tf.reset_default_graph()
# y_true = tf.Variable(labels)
y_pred = tf.Variable(predictions)
# micro, macro, weighted = tf_f1_score(y_true, y_pred)
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)
#     mic, mac, wei = sess.run([micro, macro, weighted])
    countZeros0 = tf.count_nonzero(y_pred, axis = 0)
    countZerosNone = tf.count_nonzero(y_pred, axis = None)
    
    zeroval, noneval = sess.run([countZeros0, countZerosNone])
    
    print(zeroval)
    print(noneval)
    
    


# In[78]:


#understanding tf.one_hot
d = 7 # depth is the total number of classes you have 
classLabels =  np.array([2,3,5,6,5,4,2,0,0])# the indexes 
print(classLabels.shape)
# tf.one_hot(classLabels, d)

tf.reset_default_graph()
output__ =tf.one_hot(classLabels, d)
with tf.Session() as sess:
    tf.global_variables_initializer().run(session=sess)

    
    output__val = sess.run([output__])
    
    print(output__val)
    print(output__val[0].shape)
    


# In[80]:


### This function was not done by Ali or Sean: 
# it was found at the link https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
# it calculates multi - label class F1 score and True positives, True Negs, etc.
# very useful function we modified it for our own needs 
# though since we are using a regular macro F1 score 

def tf_f1_score(y_true, y_pred):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
            weighted from the support of each class


    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true

    Returns:
        tuple(Tensor): (micro, macro, weighted)
                    tuple of the computed f1 scores
    """

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted

# tf.reset_default_graph()
# y_true = tf.Variable(labels)
# y_pred = tf.Variable(predictions)
# micro, macro, weighted = tf_f1_score(y_true, y_pred)
# with tf.Session() as sess:
#     tf.global_variables_initializer().run(session=sess)
#     mic, mac, wei = sess.run([micro, macro, weighted])
#     print(mic)
#     print(mac)
#     print(wei)


# # Restoring the Model

# In[258]:


filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model2/model_12_7__"+str(49)+"_"+str(60)+".ckpt"

graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:
        
        #define the graph
        X, Y = create_placeholders(64, 64, 26, 34)
        parameters = initialize_parameters()
        Z3 = forward_prop(X, parameters)
        #optimization 
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)

        #calculating the accuracy of the model 
        softmaxZ3 = tf.nn.softmax(Z3) # size will be 2200, 34
        output_class = tf.argmax(softmaxZ3,1) # size will be 2200, 1 or just (2200,)
        num_correct = tf.equal(output_class, tf.argmax(Y,1)) # must compare 2200
        num_correct_to_int = tf.cast(num_correct, tf.float32)
        accuracy = tf.reduce_mean(num_correct_to_int)
        
        
        saver = tf.train.Saver()

        saver.restore(sess , filepath)
        print("Model restored.")
        
    


# In[144]:


len(devMiniBatches)
len(minibatches)
len(testMiniBatches)


# # Restore the Graph and Calculate the Test Set Accuracy / F1 Score

# In[132]:


filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model2/model_12_7__"+str(49)+"_"+str(60)+".ckpt"
# dev_test_or_train = 1
# if(dev_test_or_train ==1): #train
costs
filepathEvalMetrics = "/home/muhammadayub/Desktop/CS230/models_saved/model2/model_12_7__"+str(49)+"_"+str(60)+"_evalMetrics_train.npy"
accuracies= []
minibatches_to_test_accuracy_of = copy.deepcopy(devMiniBatches)  # or devMiniBatches or testMiniBatches for test error 

print(" Running ", len(minibatches_to_test_accuracy_of) , " minibatches to get the evaluation metrics")
output_classes = []
predicted_classes = []

preds_one_hot_ev_value = None
output_one_hot_ev_value =None 


graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:
        
        #define the graph
        X, Y = create_placeholders(64, 64, 26, 34)
        parameters = initialize_parameters()
        Z3 = forward_prop(X, parameters)
        #optimization 
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)

        #calculating the accuracy of the model 
        softmaxZ3 = tf.nn.softmax(Z3) # size will be 2200, 34
        output_class = tf.argmax(softmaxZ3,1) # size will be 2200, 1 or just (2200,)  # this is the predictions vector
        output_class_of_data =  tf.argmax(Y,1)
        num_correct = tf.equal(output_class, output_class_of_data) # must compare 2200
        num_correct_to_int = tf.cast(num_correct, tf.float32)
        accuracy = tf.reduce_mean(num_correct_to_int)
        
        
        saver = tf.train.Saver()

        saver.restore(sess , filepath)
        print("Model restored.")
        
        np.random.shuffle(minibatches_to_test_accuracy_of)
        
        #devMiniBatches 
        #testMiniBatches
        for i_ , minibatch in enumerate(minibatches_to_test_accuracy_of):
            inputImage, output_image = generateMinibatch(minibatch)
            inputImage64, output_image64 = transformTo64(inputImage, output_image,34)
            output_image64 = output_image64[WaterWayMask]
            inputImage64 = inputImage64[WaterWayMask]
            #temp_cost = sess.run([cost], feed_dict={X: inputImage64, Y: output_image64})
            predicted_classes_val, output_class_val, accuracy_val = sess.run([output_class, output_class_of_data, accuracy], feed_dict={X: inputImage64, Y: output_image64})
            accuracies.append(accuracy_val)
            output_classes.append(output_class_val)
            predicted_classes.append(predicted_classes_val)
            
#             print(output_class_val.shape) #wrapped in list 
#             print(predicted_classes_val.shape)
            print('Minibatch: ', str(i_), " ", datetime.datetime.now())
            
            
#new graph for concatenating things and getting into the right format for the F1 score 
graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:            
    
        val1 = tf.concat(output_classes, axis = 0)
        
    
        #after running the minibatches-> we can concatenate the values 
        preds = tf.concat(predicted_classes,axis = 0)
        data_outputs = tf.concat(output_classes,axis = 0)
        
        d = 35 # d is the total number of classes you have 
        #classLabels =  np.array([2,3,5,6,5,4,2,0,0])# the indexes 
        # tf.one_hot(classLabels, d)

        preds_one_hot = tf.one_hot(preds, d)
        data_outputs_one_hot =tf.one_hot(data_outputs, d)#evaluated 
        preds_one_hot_ev, output_one_hot_ev = sess.run([preds_one_hot, data_outputs_one_hot])
        
        preds_one_hot_ev_value =preds_one_hot_ev
        output_one_hot_ev_value =output_one_hot_ev
        
#For Debugging below:
#         zVal = sess.run([Z3], feed_dict={X: inputImage64, Y: output_image64})
#         softMaxVal = sess.run([softmaxZ3], feed_dict={X: inputImage64, Y: output_image64})
#         output_class_val = sess.run([output_class], feed_dict={X: inputImage64, Y: output_image64})
#         num_correct_val = sess.run([num_correct], feed_dict={X: inputImage64, Y: output_image64})
#         num_correct_to_int_val = sess.run([num_correct_to_int], feed_dict={X: inputImage64, Y: output_image64})


# In[141]:


_y_true_ = None
_y_pred_ = None 
_f1Score_ = None 
_precision_ = None 
_recall_ =None 
TPVal = None
FPVal = None
FNVal = None 
_f1_ = None 
#using the F1 function found online -> get the values for the predictions  
graph = tf.Graph()
with graph.as_default(): 
    y_true = tf.Variable(output_one_hot_ev_value)
    y_pred = tf.Variable(preds_one_hot_ev_value)
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    #moded from -> https://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
    TP = tf.count_nonzero(y_pred * y_true, axis=0) 
    FP = tf.count_nonzero(y_pred * (y_true - 1), axis=0) 
    FN = tf.count_nonzero((y_pred - 1) * y_true, axis=0)  # .001 for numerical stability 

    #they are integers right now and should be floats for numerical stability 
    TP = tf.cast(TP, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)
    
    
    precision = TP / (TP + FP+tf.constant(.001))
    recall = TP / (TP + FN+tf.constant(.001))
    f1 = 2 * precision * recall / (precision + recall+tf.constant(.001))

    f1Score = tf.reduce_mean(f1)

    
    with tf.Session(graph=graph) as sess:    
        tf.global_variables_initializer().run(session=sess)
        _f1_, _y_true_, _y_pred_, _f1Score_ , _precision_ ,_recall_ , TPVal,FPVal, FNVal = sess.run([f1, y_true, y_pred, f1Score, precision ,recall, TP,FP, FN ])          


# In[142]:


filepathEvalMetrics
_f1_


# In[130]:


accuracyArray = np.array(['Accuracy', sum(np.array(accuracies).flatten())/len(accuracies)*100 , 
                          '_y_true_' , str(_y_true_),
                            '_y_pred_' ,str(_y_pred_) ,
                            '_f1Score_' ,str(_f1Score_) ,
                            '_precision_' ,str(_precision_) ,
                            '_recall_' ,str(_recall_) ,
                            'TPVal' ,str(TPVal),
                            'FPVal' , str(FPVal),
                            'FNVal' , str(FNVal)
                          ])
# we are only interested in the F1 score at the Macro level
print('F1 score (multiclass): ' , mac)        
print('Accuracy', sum(np.array(accuracies).flatten())/len(accuracies)*100)


# In[ ]:


#write the values out to the numpy file 
print(accuracyArray)
np.save(filepathEvalMetrics,accuracyArray)


# In[45]:


# output_image64[:]
plt.imshow(inputImage64[10, :,:,10])
plt.show()
output_image64_intermediate = split256by256StackOnAxis(output_image)
# output_image64 = getOutputYVector(output_image64, numCats)
print(output_image64_intermediate.shape)


# In[254]:


print('Accuracy', sum(np.array(accuracies).flatten())/len(accuracies)*100)


# In[48]:


output_image64_intermediate[10].dtype

# plt.imshow(output_image64_intermediate[10])
# plt.show()


# In[148]:


print(datetime.datetime.now())
WaterWayMask = generateWaterWayMask(waterway, 400)
inputImage, output_image = generateMinibatch(minibatches[10])
# print(inputImage.dtype)
# print(output_image.dtype)
inputImage64, output_image64 = transformTo64(inputImage, output_image,34)
# print(inputImage64.dtype)
# print(output_image64.dtype)
output_image64 = output_image64[WaterWayMask]
inputImage64 = inputImage64[WaterWayMask]
# print(inputImage64.dtype)
# print(output_image64.dtype)
print(datetime.datetime.now())


# In[143]:


print(inputImage64.shape)
print(output_image64.shape)
print(inputImage64.dtype)
print(output_image64.dtype)
print(len(WaterWayMask))


# # For back up testing and everything 

# In[96]:


# print(output_image.shape)

# output = getOutputYVector(output_image,34)
# print(output.shape)
# output[0]

# result = np.zeros((len(output_image), 34))
# cCount = None 
# for i_, img in enumerate(output_image):
#     cCount = Counter(img.flatten())


# Generating Water Way Mask

# In[128]:


def generateWaterWayMask(waterway, threshold):# between 0 and 4096, need atmost 200 to be water 
    sizeOfOneMiniBatch = int(len(indices_)/MINIBATCHES_AMT)
    waterwayMask = np.zeros((sizeOfOneMiniBatch, 256,256), dtype=np.float32)
    for i_ in range(len(waterwayMask)):
        waterwayMask[i_] = waterway
        
    #now that we have the ( 200, 256,256)  (sizeOfOneMiniBatch is 200 for example)
    #we can break it up into the 64 by 64 images
    waterwayMask64 = split256by256StackOnAxis(waterwayMask)
    #from here, you count up each one of the 64 by 64 images and set to the threshold
    waterwayMaskIndices = [] # include if they are one 
    for i_ in range(len(waterwayMask64)):
        if(np.sum(waterwayMask64[i_]) <= threshold): #> would mean you have more 1's than allowed -> not tolerable 
            waterwayMaskIndices.append(i_)
    return waterwayMaskIndices

ww = np.ones((256,256))
ww[:64, (256-64):] = 0

abc = generateWaterWayMask(ww, 5)
len(abc)
print(abc)  # think it works 

# len(minibatches[10])
#stack waterway image by 


# for getting the output data (1 hot encoded vectors)

# In[95]:


def split256by256StackOnAxis(inputImg):
    imagesList = np.split(inputImg, 4, axis = 1)
    images64by64 = []

    for almostImage in imagesList:
        imagesList64by64 = np.split(almostImage, 4, axis = 2) # axis is 0, 1, 2
        for actual64by64 in imagesList64by64:
            images64by64.append(actual64by64)
    #     for i,_64 in enumerate(images64by64):
    #         print(i, _64.shape)
    return np.concatenate(images64by64, axis = 0)

def getOutputYVector(y_output_image64, numCats): # y_output_image64 is of shape (3200,64,64)  , 34 categories + -1
    result = np.zeros((len(y_output_image64), numCats), dtype=np.float32)
    cCount = None 
    for i_, img in enumerate(y_output_image64):
        cCount = Counter(img.flatten())
        result[i_][int(cCount.most_common()[0][0])] = 1
        #get the argmax for now  -1 goes to 0, 0 goes to 1, etc. until you have 33 going to 34
#         result[i_][int(cCount.most_common()[0][0])+1] = 1
    return result


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


# Testing Copy Generation of an image ( numLayers, 256, 256) to  (256, 256, numLayers) and then to  (sampleNum, 256, 256, numLayers) 

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


# In[18]:


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
    result = np.zeros(shape=(times, _3dimg_shape[0], _3dimg_shape[1],_3dimg_shape[2] ), dtype=np.float32)
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


# In[16]:


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


# In[10]:


# zipLoc = cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\drive-download-20181203T055115Z-005.zip')
# directory = cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\\')+'/'

# print(zipLoc)
# print(directory)
# with zipfile.ZipFile(zipLoc, 'r') as zip_ref:
#     zip_ref.extractall(directory)
# a = np.load(cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\x__buildings_.npy'))
# #b = cleanURL(r'C:\Users\User\Documents\CS230 Project\new_github\data_for_cnn_training\New folder\y__c16_.npy')
# #a = np.load(b)
# a.shape


# In[74]:


# inputImage, output_image, waterway = generateMinibatch(minibatches[10])
#inputImage, output_image = generateMinibatch(minibatches[10])


# In[190]:


# tf.reset_default_graph()

learning_rate = .008
costs = []
num_epochs = 1

WaterWayMask = generateWaterWayMask(waterway, 400)

graph = tf.Graph()
with graph.as_default(): # https://stackoverflow.com/questions/36281129/no-variable-to-save-error-in-tensorflow

    #variable declarations must be before tf.train.Saver() unless here: https://stackoverflow.com/questions/50974976/tensorflow-why-must-saver-tf-train-saver-be-declared-after-variables-are
    #the model
#     X, Y = create_placeholders(64, 64, 26, 34)
#     parameters = initialize_parameters()
#     Z3 = forward_prop(X, parameters)

#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
#     optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)
    v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
    v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

    inc_v1 = v1.assign(v1+1)
    dec_v2 = v2.assign(v2-1)
    
    saver = tf.train.Saver()

    # # Initialize all the variables globally
    # init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:


        #init must be after optimizer
        init = tf.global_variables_initializer()

        # Run the initialization
        sess.run(init)
        print('Starting ' , datetime.datetime.now())
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            np.random.shuffle(minibatches) # get new minibatch results 

            for i_ , minibatch in enumerate(minibatches[:30]):

                # Select a minibatch
#                 inputImage, output_image = generateMinibatch(minibatches[10])
#                 inputImage64, output_image64 = transformTo64(inputImage, output_image,34)
#                 output_image64 = output_image64[WaterWayMask]
#                 inputImage64 = inputImage64[WaterWayMask]
    
                a = sess.run(inc_v1)
                b= sess.run(dec_v2)
                print(a , " ", b)

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
#                 _ , temp_cost = sess.run([optimizer,cost], feed_dict={X: inputImage64, Y: output_image64})
                ### END CODE HERE ###

#                 minibatch_cost += temp_cost / MINIBATCHES_AMT

                print('Minibatch End: ', datetime.datetime.now())

                if(i_ % 2 == 0 ):
                    #save model
                    save_path = saver.save(sess, "/home/muhammadayub/Desktop/CS230/models_saved/model1/model_12_7__"+str(epoch)+"_"+str(i_)+".ckpt")



            costs.append(minibatch_cost)
            print('Epoch End: ', datetime.datetime.now())



# In[189]:


graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:
        #define the graph
#         X, Y = create_placeholders(64, 64, 26, 34)
#         parameters = initialize_parameters()
#         Z3 = forward_prop(X, parameters)

#         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
#         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)
        v1 = tf.get_variable("v1", shape=[3])
        v2 = tf.get_variable("v2", shape=[5])


        saver = tf.train.Saver()
#         saver.restore(sess, "/home/muhammadayub/Desktop/CS230/models_saved/model_12_7__0_2.ckpt")
        saver.restore(sess , "/home/muhammadayub/Desktop/CS230/models_saved/model1/model_12_7__"+str(0)+"_"+str(28)+".ckpt")
        print("Model restored.")
#         temp_cost = sess.run([cost], feed_dict={X: inputImage64, Y: output_image64})
#         print(temp_cost)
        print("v1 : %s" % v1.eval())
        print("v2 : %s" % v2.eval())

        


# In[201]:


# tf.reset_default_graph()

learning_rate = .008
costs = []
num_epochs = 1

WaterWayMask = generateWaterWayMask(waterway, 400)

graph = tf.Graph()
with graph.as_default(): # https://stackoverflow.com/questions/36281129/no-variable-to-save-error-in-tensorflow

    #variable declarations must be before tf.train.Saver() unless here: https://stackoverflow.com/questions/50974976/tensorflow-why-must-saver-tf-train-saver-be-declared-after-variables-are
    #the model
    X, Y = create_placeholders(64, 64, 26, 34)
    parameters = initialize_parameters()
    Z3 = forward_prop(X, parameters)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)

    
    saver = tf.train.Saver()

    # # Initialize all the variables globally
    # init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:


        #init must be after optimizer
        init = tf.global_variables_initializer()

        # Run the initialization
        sess.run(init)
        print('Starting ' , datetime.datetime.now())
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            np.random.shuffle(minibatches) # get new minibatch results 

            for i_ , minibatch in enumerate(minibatches[:4]):

                # Select a minibatch
                inputImage, output_image = generateMinibatch(minibatches[10])
                inputImage64, output_image64 = transformTo64(inputImage, output_image,34)
                output_image64 = output_image64[WaterWayMask]
                inputImage64 = inputImage64[WaterWayMask]


                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer,cost], feed_dict={X: inputImage64, Y: output_image64})
                ### END CODE HERE ###

                minibatch_cost += temp_cost / MINIBATCHES_AMT

                print('Minibatch End: ', datetime.datetime.now())
                print(temp_cost,' ' , epoch,' ' ,i_)
                if((epoch ==0) and (i_==0)):
                    globalA = inputImage64
                    globalB = output_image64
                
                if(i_ % 2 == 0 ):
                    #save model
                    filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model1/model_12_7__"+str(epoch)+"_"+str(i_)+".ckpt"
                    print(temp_cost, " at file name: ", filepath[25:])
                    save_path = saver.save(sess, filepath)

            costs.append(minibatch_cost)
            print('Epoch End: ', datetime.datetime.now())



# # Restoring the Model

# In[202]:


graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:
        #define the graph
        X, Y = create_placeholders(64, 64, 26, 34)
        parameters = initialize_parameters()
        Z3 = forward_prop(X, parameters)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)



        saver = tf.train.Saver()
        filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model1/model_12_7__"+str(0)+"_"+str(0)+".ckpt"
        saver.restore(sess , filepath)
        print("Model restored.")
#         temp_cost = sess.run([cost], feed_dict={X: inputImage64, Y: output_image64})
        temp_cost = sess.run([cost], feed_dict={X: globalA, Y: globalB})
        print(temp_cost)

        


# In[244]:


graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:
        #define the graph
        X, Y = create_placeholders(64, 64, 26, 34)
        parameters = initialize_parameters()
        Z3 = forward_prop(X, parameters)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)

        #calculating the accuracy of the model 
        softmaxZ3 = tf.nn.softmax(Z3) # size will be 2200, 34
        output_class = tf.argmax(softmaxZ3,1) # size will be 2200, 1 or just (2200,)
        num_correct = tf.equal(output_class, tf.argmax(Y,1)) # must compare 2200
        num_correct_to_int = tf.cast(num_correct, tf.float32)
        accuracy = tf.reduce_mean(num_correct_to_int)
        
        
        saver = tf.train.Saver()
        filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model1/model_12_7__"+str(0)+"_"+str(0)+".ckpt"
        saver.restore(sess , filepath)
        print("Model restored.")
#         temp_cost = sess.run([cost], feed_dict={X: inputImage64, Y: output_image64})
        temp_cost = sess.run([cost], feed_dict={X: globalA, Y: globalB})
        zVal = sess.run([Z3], feed_dict={X: globalA, Y: globalB})
        softMaxVal = sess.run([softmaxZ3], feed_dict={X: globalA, Y: globalB})
        output_class_val = sess.run([output_class], feed_dict={X: globalA, Y: globalB})
        num_correct_val = sess.run([num_correct], feed_dict={X: globalA, Y: globalB})
        num_correct_to_int_val = sess.run([num_correct_to_int], feed_dict={X: globalA, Y: globalB})
        accuracy_val = sess.run([accuracy], feed_dict={X: globalA, Y: globalB})
        
        random = sess.run([tf.argmax(Y,1)], feed_dict={X: globalA, Y: globalB})
        
        print(len(zVal))
        print(zVal[0].shape)
        print(softMaxVal[0].shape)
        print(zVal[0][0])
        print(softMaxVal[0][0])
        print(output_class_val[0].shape)
        print(output_class_val[0])
        print(temp_cost)
        print('num_correct_val: ', num_correct_val[0][:12])
        print(random)
        print('num_correct_to_int_val ', num_correct_to_int_val[0][:12])
        print('accuracy_val ', accuracy_val)
        


# In[245]:


print(globalB.shape)
set(np.where(globalB == 1)[1].tolist())  
globalB[10][0] = 0
globalB[10][5] = 1
globalB[:12]
print(2199/2200)


# In[220]:


graph = tf.Graph()
with graph.as_default(): 
    with tf.Session(graph=graph) as sess:
        #define the graph
        X, Y = create_placeholders(64, 64, 26, 34)
        parameters = initialize_parameters()
        Z3 = forward_prop(X, parameters)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)#1e-4).minimize(cross_entropy)

        softmaxZ3 = tf.nn.softmax(Z3)

        saver = tf.train.Saver()
        filepath = "/home/muhammadayub/Desktop/CS230/models_saved/model1/model_12_7__"+str(0)+"_"+str(0)+".ckpt"
        saver.restore(sess , filepath)
        print("Model restored.")
#         temp_cost = sess.run([cost], feed_dict={X: inputImage64, Y: output_image64})
        temp_cost = sess.run([cost], feed_dict={X: globalA, Y: globalB})
        zVal = sess.run([Z3], feed_dict={X: globalA, Y: globalB})
        print(len(zVal))
        print(zVal[0].shape)
        
        print(temp_cost)

        


# In[ ]:



mic = None 
mac = None 
wei = None        
#using the F1 function found online -> get the values for the predictions  
graph = tf.Graph()
with graph.as_default(): 
    y_true = tf.Variable(output_one_hot_ev_value)
    y_pred = tf.Variable(preds_one_hot_ev_value)
    micro, macro, weighted = tf_f1_score(y_true, y_pred)
    
    with tf.Session(graph=graph) as sess:    
        tf.global_variables_initializer().run(session=sess)
        mic, mac, wei = sess.run([micro, macro, weighted])  

