# Fighting Crime with Deep Learning 

*Authors: Sean Fitzgerald and Muhammad Ayub

We use Tensorflow for programming the networks. Take a look at some [tutorials] we found(https://github.com/easy-tensorflow/easy-tensorflow).


## Requirements to Run Code

Please have Python 3 installed. 
For running Sean's code the libraries needed are in cs230.yml.
 
For running Ali's code the libraries are in requirements.txt. Ali used Windows 7 OS. 
Also, Python 3.5 was used with a conda virtual environment. 
With conda installed make a new environment and then do : 

```
activate py35
pip install -r requirements.txt   
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Sean's CNN predicts crime volume. I.e. - given a 64x64xnum_channels image, predict the number of crimes that occur. 

Ali's CNN classifies crime type. I.e. - given a 64x64xnum_channels image, predict the most likely class. 

## Download the Datasets

For crime volume prediction, please refer to Sean's section to download the dataset. 

For crime classification, please refer to Ali's section to download the dataset. Specifically the CNN data is hosted and available at: 
 https://drive.google.com/drive/folders/1LOyrqORS4Zszm4n62siat888IIgpx4rx?usp=sharing. 
You will find a text file detailing how the data is structured. 

```
Final Folder Public Datasets/
    inputs_data/
        input_0_.npy
		input_1_.npy
        ...
    outputs_data/
        output_0_.npy
		output_1_.npy
        ...
	CNN Dataset explained.txt
```

Each .npy file is a minibatch. You are free to concatenate minibatches to get the whole dataset or a subset. 
For example input_0_.npy will have 200 images of 64x64xnum_channels . The output_0_.npy file will have 200 64x64 images of the crime locations. 


```
please see the SampleData to view what the images look like. Most of the images are layers/channels in a single input datapoint. 
```


## Quickstart (Volume of Crime Model)

Please take a look at Sean's sub folder and go to Prediction Notebooks folder as well as Data Configure Notebooks folder. 

## Quickstart (Classification Model)


1. Download the dataset and get a feel for the dataset. Layers information can be found at the datasources website. 
Please take a look at Ali/cnn_data_engineering_related_new to understand how raw data in the Google Drive folder is turned into input and output images


2.  The output data are still images so you can convert them to softmax vectors for classification or scalars for some sort of regression (you could add up all the non -1 values to get 
total count and hence you can train networks for regression). Anyways, the general format is that you can read in the input/output data, preprocess it and then run the model. 

3. The files that start with " CNN Run and Restore to Run " are the files that are for training the neural network. You can specify where and how often you want to save the .cpkt model files to disk.
To train, you must have about 64 GB of RAM. If you don't you can simply read in each input/ouput.npy file , convert the output 64x64 to your desired format (a softmax vector or a scalar value for regression), 
save these values to disk and then just load in one minibatch (one input, and one processed output file) before feeding into network. 

```
outputMinibatch5 = np.load('/output_5_.npy')
softmax_one_hot_encoded  = convertToOneHot(outputMinibatch5)
#as an example, write the output_5_.npy file to disk converting it to output_5_onehot.npy
np.save('/output_5_onehot.npy', softmax_one_hot_encoded)

# then later on, just uste the 'output_5_onehot.npy' file to feed in a minibatch
inputData = np.load('/input_5_.npy' , 'r')
outputData = np.load('/output_5_onehot.npy', 'r')
...
sess.run(feed_dict = {X: inputData, Y: outputData}

```

4. Pleas go to Ali/notebooks_models/ to see the code files. 

Model 5a used one hot encoding. Learning rate was .008 , minibatch size was 200.

Model 5b used 3 hot encoding. Learning rate was .008, minibatch size was 200. 

Model 5c used 3 hot encoding. Learning rate was .003, minibatch size was 400. 

Model 6 used 3 hot encoding with upsampling. Learning rate was .003, minibatch size was 400. 


## Raw Data/ Resources

Please contact babaali9966@gmail.com for help


The raw data we used was from the data.gov website. 
The Crime data is located at: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
We joined in a bunch of other data like socioeconomic-indicators-in-chicago-2008-2012-36e55: https://catalog.data.gov/dataset/census-data-selected-socioeconomic-indicators-in-chicago-2008-2012-36e55
