import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pandas as pd
import math
import json
import os
import time
# For EC2
import boto3
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

################################
# NON-HYPERPARAMETER CONSTANTS #
################################
final_dataset_location = '/Volumes/GoogleDrive/My Drive/Crime Data/Final Folder/'
playground_dataset_location = '/Volumes/GoogleDrive/My Drive/Crime Data/Composute Data/Sean Workspace/'
# trial_file_location = '/Users/sean/Documents/Education/Stanford/230/Project/Sean/Trials/'
trial_file_location = '/home/ec2-user/cs230/crime_prediction/Sean/Trials/'
# pickled_model_location = '/Users/sean/Documents/Education/Stanford/230/Project/Sean/Trials/Pickled Models/Trial %d.ckpt'
pickled_model_location = '/home/ec2-user/cs230/crime_prediction/Sean/Trials/Pickled Models/Trial %d.ckpt'
trial_file_format = 'Trial %d.xlsx'
epochs_between_prints = 100
hyperparameter_file_columns = ['Epoch Cost',
                               'Train Accuracy',
                               'Dev Accuracy',
                               'Duration',
                               'Dev Set Proportion',
                               'Test Set Proportion',
                               'Train Set Proportion',
                               'Learning Rate',
                               'Goal Total Epochs',
                               'Minibatch Size',
                               'Hidden Units per Layer',
                               'Hidden Layers',
                               'Dataset',
                               'Optimizer Name']

###################
# HYPERPARAMETERS #
###################
np.random.seed(0)
dev_set_proportion = 0.01
test_set_proportion = 0.01
train_set_proportion = 1 - (dev_set_proportion + test_set_proportion)
learning_rate = 0.001
goal_total_epochs = 10000
minibatch_size = np.inf
hidden_units_per_layer = 100
num_hidden_layers = 14
trial_number = 33
dataset = "19_November.xlsx"
optimizer_name = 'Adam'

print('Loading Crime Dataset')

# For EC2
bucket = "cs230"
file_name = "19_November.xlsx"

s3 = boto3.client('s3') 
# 's3' is a key word. create connection to S3 using default config and all buckets within S3

obj = s3.get_object(Bucket= bucket, Key= file_name) 
# get object and file (key) from bucket

crime_data = pd.read_excel(obj['Body']) # 'Body' is a key word

# For Local Machine
# crime_data = pd.read_excel(final_dataset_location + dataset)

# Utility Functions #

####################
# EPOCH MANAGEMENT #
####################

def restore_model(saver, session):
    # Before epoch, check for trial # in trial files
    if os.path.isfile(trial_file_location+trial_file_format % trial_number):
        print('Model found.  Restoring parameters.')
        # If trial exists:
        # 1. roll back (cost, train & dev accuracy) to epoch with highest dev accuracy.
        trial_hyperparameters = pd.read_excel(trial_file_location+trial_file_format % trial_number)
        # Find highest dev accuracy
        best_dev_index = np.argmax(trial_hyperparameters.loc[:,'Dev Accuracy'].values)
        # Delete all rows after this epoch
        trial_hyperparameters = trial_hyperparameters[:best_dev_index+1]
        # 2. restore model for the best dev accuracy
        saver.restore(session, pickled_model_location % trial_number)
        # Save the edited/new hyperparameter trial file
        writer = pd.ExcelWriter(trial_file_location+trial_file_format % trial_number)
        trial_hyperparameters.to_excel(writer)
        writer.save()
        # Return the number of epochs already trained
        return len(trial_hyperparameters)
    else:
        print('No saved model.  Using default parameter initialization.')
        return 0

def epoch_teardown(saver, session, cost, training_accuracy, dev_accuracy, duration):
    trial_hyperparameters = pd.DataFrame(columns=hyperparameter_file_columns)
    # After epoch, check for hyperparameter file
    if os.path.isfile(trial_file_location+trial_file_format % trial_number):
        trial_hyperparameters = pd.read_excel(trial_file_location+trial_file_format % trial_number)
        # Compare dev accuracy with all other epochs
        max_dev_accuracy = np.max(trial_hyperparameters['Dev Accuracy'].values)
        if dev_accuracy > max_dev_accuracy:
            # If greatest, save model
            saver.save(session, pickled_model_location % trial_number)
    # Save hyperparameters, epoch cost, and training & dev accuracies
    trial_hyperparameters = trial_hyperparameters.append({
        'Epoch Cost' : cost,
        'Train Accuracy' : training_accuracy,
        'Dev Accuracy' : dev_accuracy,
        'Duration' : duration,
        'Dev Set Proportion' : dev_set_proportion,
        'Test Set Proportion' : test_set_proportion,
        'Train Set Proportion' : train_set_proportion,
        'Learning Rate' : learning_rate,
        'Goal Total Epochs' : goal_total_epochs,
        'Minibatch Size' : minibatch_size,
        'Hidden Units per Layer' : hidden_units_per_layer,
        'Hidden Layers' : num_hidden_layers,
        'Dataset' : dataset,
        'Optimizer Name' : optimizer_name
    }, ignore_index=True)
    # Save the edited/new hyperparameter trial file
    writer = pd.ExcelWriter(trial_file_location+trial_file_format % trial_number)
    trial_hyperparameters.to_excel(writer)
    writer.save()

def random_mini_batches(X, Y, mini_batch_size = 64):
    # Creates a list of random minibatches from (X, Y)
    m = X.shape[1]
    mini_batches = []
    
    if mini_batch_size > m:
        mini_batches.append((X,Y))
    else:
        # Step 1: Shuffle (X, Y)
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((1,m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k*mini_batch_size: (k+1)*(mini_batch_size)]
            mini_batch_Y = shuffled_Y[:, k*mini_batch_size: (k+1)*(mini_batch_size)]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, int(mini_batch_size*np.floor(m/mini_batch_size)): m]
            mini_batch_Y = shuffled_Y[:, int(mini_batch_size*np.floor(m/mini_batch_size)): m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
    
    return mini_batches

###################################
# CREATE NEURAL NETWORK STRUCTURE #
###################################

def create_NN_structure(n_x, n_y):
    ops.reset_default_graph()

    # Create placeholders for the featuers and labels
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.int32, shape=(n_y, None), name='Y')

    # Create the network parameters
    parameters = {}
    for layer in range(num_hidden_layers+1):
        previous_layer_size = (n_x if layer == 0 else hidden_units_per_layer)
        this_layer_size = (n_y if layer == num_hidden_layers else hidden_units_per_layer)
        W_name = 'W'+str(layer+1)
        b_name = 'b'+str(layer+1)
        parameters[W_name] = tf.get_variable(W_name,
                                             (this_layer_size,previous_layer_size),
                                             initializer=tf.contrib.layers.xavier_initializer(seed=1, uniform=False))
        parameters[b_name] = tf.get_variable(b_name,
                                             (this_layer_size,1),
                                             initializer=tf.zeros_initializer())

    # Hook up the network layers
    A = X
    Z = X
    for layer in range(num_hidden_layers+1):
        W = parameters['W'+str(layer+1)]
        b = parameters['b'+str(layer+1)]
        Z = W@A+b
        A = tf.nn.relu(Z)
    Z_hat = Z
    Y_hat = tf.argmax(tf.transpose(tf.nn.softmax(tf.transpose(Z_hat))), axis=0)
    
    return Z_hat, Y_hat, X, Y

#############################
# CREATE AND CONDITION DATA #
#############################

def create_and_condition_data():
    # Convert the dataframe to numpy arrays for features and labels
    features = crime_data.drop(columns=['categoryCode','closestGroceryStore']).values.T
    labels = crime_data.loc[:,'categoryCode'].values.reshape((-1,1)).T
    # Drop all NAs that were caught in the transfer
    feature_cols_with_nans = np.isnan(features).any(axis=0)
    features = features[:,~feature_cols_with_nans]
    labels = labels[:,~feature_cols_with_nans]
    label_cols_with_nans = np.isnan(labels).any(axis=0)
    features = features[:,~label_cols_with_nans]
    labels = labels[:,~label_cols_with_nans]

    n_x, m = features.shape
    n_y = len(crime_data.loc[:,'categoryCode'].unique())

    # Shuffle the data
    order = np.argsort(np.random.random(m))
    features = features[:,order]
    labels = labels[:,order]

    # One Hot Encode the Labels
    one_hot = np.zeros((n_y,m))
    one_hot[labels,np.arange(m)] = 1
    labels = one_hot

    # Split between train, dev, and test
    # Data structure: [     TRAIN     ][ DEV ][ TEST ]
    dev_start_index = int(train_set_proportion*m)
    test_start_index = dev_start_index + int(dev_set_proportion*m)

    X_train = features[:, 0:dev_start_index]
    Y_train = labels[:, 0:dev_start_index]

    X_dev = features[:, dev_start_index:test_start_index]
    Y_dev = labels[:, dev_start_index:test_start_index]

    X_test = features[:, test_start_index:]
    Y_test = labels[:, test_start_index:]

    # Normalize the inputs and outputs based on the training set mean and variance
    x_mean = X_train.mean(axis=1).reshape(n_x,1)
    x_variance = X_train.var(axis=1).reshape(n_x,1)

    X_train = (X_train-x_mean)/x_variance
    X_dev = (X_dev-x_mean)/x_variance
    X_test = (X_test-x_mean)/x_variance
    
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

#################
# EXECUTE MODEL #
#################

def execute_model():
    global optimizer_name, trial_file_location

    print('Conditioning Data')
    X_train, Y_train, X_dev, Y_dev, X_test, Y_test = create_and_condition_data()
    n_x, m = X_train.shape
    n_y, _ = Y_train.shape
    print('Creating Network Structure')
    Z_hat, Y_hat, X, Y = create_NN_structure(n_x, n_y)

    # Calculate the cost from the network prediction
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.transpose(Z_hat),
                                                                     labels=tf.transpose(Y)))
    optimizer = None
    # Create the optimizer
    if optimizer_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    else:
        optimizer_name = 'GD'
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Formula for calculating set accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Z_hat), tf.argmax(Y)), "float"))

    # Run the tf session to train and test
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        # If the trial already exists, pick up where we left off
        starting_epoch = restore_model(saver, session)
        print('Beginning Training')
        for epoch in range(starting_epoch, goal_total_epochs):
            start_time = time.time()
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            if num_minibatches < 1: num_minibatches=1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = session.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            elapsed_time = time.time() - start_time

            # Display epoch results every so often
            if epoch % epochs_between_prints == 0:
                print('%i Epochs' % epoch)
                print('\tCost: %f' % epoch_cost)
                print('\tTrain Accuracy: %f' % accuracy.eval({X: X_train, Y: Y_train}))
                print('\tDev Accuracy: %f' % accuracy.eval({X: X_dev, Y: Y_dev}))

            # Epoch over, tear down
            epoch_teardown(saver,
                           session,
                           epoch_cost,
                           float(accuracy.eval({X: X_train, Y: Y_train})),
                           float(accuracy.eval({X: X_dev, Y: Y_dev})),
                           elapsed_time)

        # Calculate the accuracy on the train and dev sets
        print('Reached Goal Number of Epochs.')
        print('Final Train Accuracy: %f' % accuracy.eval({X: X_train, Y: Y_train}))
        print('Final Dev Accuracy: %f' % accuracy.eval({X: X_dev, Y: Y_dev}))


execute_model()
