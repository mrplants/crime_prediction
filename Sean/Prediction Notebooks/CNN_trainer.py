import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow.python.framework import ops
import pandas as pd
import math
import json
import os
import time
from datetime import datetime
import datetime as dtm
import random
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
USING_EC2 = True
# For EC2
if USING_EC2:
    import boto3
    from io import BytesIO

###################
# HYPERPARAMETERS #
###################
np.random.seed(0)

learning_rate = 0.06
goal_total_epochs = 1000
trial_number = 13
optimizer_name = 'Adam'
regular_lambda = 0
accuracy_eval_batch_size = 1000
max_num_crimes = 20
minibatch_size = 5000
num_minibatches = 5
output_grid_width = 1 # x position
output_grid_height = output_grid_width # y position
num_anchor_boxes = 5
input_image_width = 64
input_image_height = 64
test_set_size = 1000
dev_set_size = 1000
network_description = 'CONV(f=4,s=1,"same")x8 => MAX-POOL(f=8) => CONV(f=4,s=1,"same")x16 => MAX-POOL(f=4) => FC(100) => FC(100) => SOFT(NUM_CRIMES)'
num_samples_for_normalization = 1000

################################
# NON-HYPERPARAMETER CONSTANTS #
################################
# For EC2
if USING_EC2:
    bucket = 'cs230'
    s3 = boto3.client('s3')
processed_dataset_paths_xlsx = '/Volumes/GoogleDrive/My Drive/Crime Data/Composite Data/Sean Workspace/Processed/%s.xlsx' 
dataset_location = '/Volumes/GoogleDrive/My Drive/Crime Data/Composite Data/Sean Workspace/CNN Final/'
trial_file_location = '/Users/sean/Documents/Education/Stanford/230/Project/Sean/Trials/'
AWS_trial_file_location = '/home/ec2-user/cs230/crime_prediction/Sean/Trials/'
if USING_EC2:
    trial_file_location = AWS_trial_file_location
pickled_model_location = '/Users/sean/Documents/Education/Stanford/230/Project/Sean/Trials/Pickled Models/CNN Trial %d.ckpt'
AWS_pickled_model_location = '/home/ec2-user/cs230/crime_prediction/Sean/Trials/Pickled Models/CNN Trial %d.ckpt'
if USING_EC2:
    pickled_model_location = AWS_pickled_model_location
trial_file_format = 'CNN Trial %d.xlsx'
epochs_between_prints = 1
hyperparameter_file_columns = [
    'Training Cost',
    'Dev Cost',
    'Accuracy Evaluation Batch Size',
    'Maximum Crimes per Window',
    'Number of Anchor Boxes per Window',
    'Input Image Width',
    'Input Image Height',
    'Output Grid Width',
    'Output Grid Height',
    'Train Accuracy',
    'Dev Accuracy',
    'Duration',
    'Learning Rate',
    'Goal Total Epochs',
    'Minibatch Size',
    'Number of Minibatches',
    'Optimizer Name',
    'L2 Regularization Lambda',
    'Test Set Size',
    'Dev Set Size',
    'Network Description',
    'num_samples_for_normalization']
FIRST_DATE = datetime(2001, 1, 1)
LAST_DATE = datetime(2018, 1, 1)
NUM_DAYS = (LAST_DATE-FIRST_DATE).days
# 25 channels + date channels (17+12+31+6) = 91
X_MAX_PIXELS = 2048
Y_MAX_PIXELS = X_MAX_PIXELS
X_WINDOW_MAX_PIXELS = input_image_width
Y_WINDOW_MAX_PIXELS = input_image_height
X_HALF_WINDOW_PIXELS = int(X_WINDOW_MAX_PIXELS/2)
Y_HALF_WINDOW_PIXELS = X_HALF_WINDOW_PIXELS
NUM_STATIC_CHANNELS = 28
STREET_CHANNEL, WATERWAY_CHANNEL, PARK_CHANNEL, FOREST_CHANNEL, SCHOOL_CHANNEL, LIBRARY_CHANNEL, BUILDING_CHANNELS,_,_,_,_,_,_,_,_,_, BUSINESS_CHANNELS,_,_,_,_, SOCIO_CHANNELS,_,_,_,_,_,_ = range(NUM_STATIC_CHANNELS)
NUM_DYNAMIC_CHANNELS = 9#12
LIFE_EXPECTANCY_CHANNEL, L_CHANNELS,_,_,_,_,_,_,_ = range(NUM_STATIC_CHANNELS,NUM_STATIC_CHANNELS+NUM_DYNAMIC_CHANNELS)
NUM_TIME_SLOTS = 12
NUM_INPUT_CHANNELS = NUM_STATIC_CHANNELS+NUM_DYNAMIC_CHANNELS
L_LINES = ['Green','Red','Brown','Purple','Yellow','Blue','Pink','Orange']
CRIME_CATEGORIES = ['BATTERY', 'OTHER OFFENSE', 'ROBBERY', 'NARCOTICS', 'CRIMINAL DAMAGE',
                    'WEAPONS VIOLATION', 'THEFT', 'BURGLARY', 'MOTOR VEHICLE THEFT',
                    'PUBLIC PEACE VIOLATION', 'ASSAULT', 'CRIMINAL TRESPASS',
                    'CRIM SEXUAL ASSAULT', 'INTERFERENCE WITH PUBLIC OFFICER', 'ARSON',
                    'DECEPTIVE PRACTICE', 'LIQUOR LAW VIOLATION', 'KIDNAPPING',
                    'SEX OFFENSE', 'OFFENSE INVOLVING CHILDREN', 'PROSTITUTION', 'HOMICIDE',
                    'GAMBLING', 'INTIMIDATION', 'STALKING', 'OBSCENITY', 'PUBLIC INDECENCY',
                    'HUMAN TRAFFICKING', 'CONCEALED CARRY LICENSE VIOLATION',
                    'OTHER NARCOTIC VIOLATION', 'NON - CRIMINAL', 'NON-CRIMINAL',
                    'NON-CRIMINAL (SUBJECT SPECIFIED)', 'RITUALISM', 'DOMESTIC VIOLENCE']

#####################################
# DATA PROCESSING UTILITY FUNCTIONS #
#####################################

def extract_data_for_date(record, fast_lookup, column):
    record_date = datetime(record.Date.year, record.Date.month, record.Date.day)
    index = (record_date - FIRST_DATE).days
    if index < NUM_DAYS and index >= 0:
        fast_lookup[index] = record[column]

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    Reference: https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

######################
# DATAFRAME CREATION #
######################

# Building CNN Data
# Data that varies with Time and Location:
# - Crime (OUTPUT - YOLO with crime and location)
# - L entries (8 layers - one per line)
# - Life Expectancy (1 layer)
#
# Data that varies with Time Only:
# - Weather (3 layers - MIN TEMP, MAX TEMP, and PRECIPITATION)
# - Date
# - Time
#
# Data that varies with Location Only:
# - Businesses (5 layers - types of businesses)
# - Buildings (10 layers - stories|units|sqfeet for sound|minor repair|major repair.  Also uninhabitable or not.)
# - Waterways (1 layer)
# - Major Streets (1 layer)
# - Libraries (1 layer)
# - Public Parks (1 layer)
# - Forests (1 layer)
# - Schools (1 layer)
# - Socioeconomic channels (7 socioeconomic indicators)

def normalize_input_data(layer_means, layer_variances):
    global normalized_zeros, normalized_static_channels, normalized_min_temp_lookup, normalized_max_temp_lookup
    global normalized_precipitation_lookup, normalized_life_expectancy_frame, normalized_L_entries_compressed
    global normalized_months, normalized_days, normalized_time_slots, INPUT_DATA_NORMALIZED
    
    # Calculate the normalized zero with which to fill the minibatch inputs
    zero_variance_locations = np.where(layer_variances == 0)
    fixed_variances = np.copy(layer_variances)
    fixed_variances[zero_variance_locations] = 1
    normalized_zeros = layer_means / fixed_variances
    # Normalize the static channels
    normalized_static_channels = ((static_channels.transpose(1,2,0) - layer_means[:NUM_STATIC_CHANNELS]) / fixed_variances[:NUM_STATIC_CHANNELS]).transpose(2,0,1)
    # Normalize the weather data
    normalized_min_temp_lookup = (min_temp_lookup - np.mean(min_temp_lookup)) / np.var(min_temp_lookup)
    normalized_max_temp_lookup = (max_temp_lookup - np.mean(max_temp_lookup)) / np.var(max_temp_lookup)
    normalized_precipitation_lookup = (precipitation_lookup - np.mean(precipitation_lookup)) / np.var(precipitation_lookup)
    # Normalize life expectancy
    normalized_life_expectancy_frame = (life_expectancy_frame - layer_means[LIFE_EXPECTANCY_CHANNEL]) / fixed_variances[LIFE_EXPECTANCY_CHANNEL]
    # Normalize L Entries
    normalized_L_entries_compressed = pd.DataFrame(columns = L_LINES)
    def normalize_L_entries(location_entries, means, variances):
        for line_index,line in enumerate(L_LINES):
            normalized_entries = np.copy(location_entries[line].astype(np.float64))
            normalized_entries[2] = (normalized_entries[2] - means[line_index]) / variances[line_index]
            normalized_L_entries_compressed.loc[location_entries.name, [line]] = [normalized_entries]
    L_entries_compressed.apply(normalize_L_entries,
                               args = (layer_means[L_CHANNELS:L_CHANNELS+len(L_LINES)], fixed_variances[L_CHANNELS:L_CHANNELS+len(L_LINES)]),
                               axis=1)
    # Calculate normalized date and time 'ones'
    INPUT_DATA_NORMALIZED = True
    
def choose_random_crime():
    # Sample randomly and uniformly from the crimes
    this_crime_index = crime_indices[np.random.randint(len(crime_indices))]
    random_day = this_crime_index[0]
    random_time_slot = this_crime_index[1]
    random_category = this_crime_index[2]
    random_crime_index = this_crime_index[4]
    return random_day, random_time_slot, random_category, random_crime_index

def generate_random_mini_batch(batch_size, avoid_these_samples, layer_means = None, layer_variances = None, normalize = True):
    # Generate samples
    return generate_mini_batch(sample_index_and_location(batch_size, avoid_these_samples),
                               layer_means = layer_means,
                               layer_variances = layer_variances,
                               normalize = normalize)

def generate_mini_batch(samples, layer_means = None, layer_variances = None, normalize = True):
    # Normalize the input data if necessary
    if not INPUT_DATA_NORMALIZED:
        if layer_means is None or layer_variances is None:
            print('Usage Error: Please normalize the input data or pass in the layer means and variances to generate_mini_batch().')
            print('Generating mini-batch with non-normalized input data.')
        elif normalize:
            normalize_input_data(layer_means, layer_variances)
    day_index_samples = samples['day_index']
    batch_size = len(day_index_samples)
    years_for_samples = year_fast_lookup[day_index_samples]
    months_for_samples = month_fast_lookup[day_index_samples]
    days_for_samples = day_fast_lookup[day_index_samples]
    time_slot_samples = samples['time_slot']
    location_x_samples = samples['location_x']
    location_y_samples = samples['location_y']

    # Fill the mini-batches with normlized zeros
    mini_batch_input = np.repeat(np.repeat(np.repeat(normalized_zeros[:,np.newaxis],
                                                     X_WINDOW_MAX_PIXELS,
                                                     axis=1)[:,:,np.newaxis],
                                           Y_WINDOW_MAX_PIXELS,
                                           axis=2)[np.newaxis,:,:,:],
                                 batch_size,
                                 axis=0)
    mini_batch_output = np.zeros((batch_size, X_WINDOW_MAX_PIXELS, Y_WINDOW_MAX_PIXELS))

    # Since indices are batched, cannot use the get_input, get_output locations above (they are single use only)
    # Add static channels first (cannot vectorize here)
    for batch_index in range(batch_size):
        mini_batch_input[batch_index,:NUM_STATIC_CHANNELS] = normalized_static_channels[:,location_x_samples[batch_index]-X_HALF_WINDOW_PIXELS:location_x_samples[batch_index]+X_HALF_WINDOW_PIXELS, location_y_samples[batch_index]-Y_HALF_WINDOW_PIXELS:location_y_samples[batch_index]+Y_HALF_WINDOW_PIXELS]
    # Life Expectancy Channel (cannot vectorize here)
    for batch_index in range(batch_size):
        mini_batch_input[batch_index,LIFE_EXPECTANCY_CHANNEL] = normalized_life_expectancy_frame[years_for_samples[batch_index]-FIRST_DATE.year, location_x_samples[batch_index]-X_HALF_WINDOW_PIXELS:location_x_samples[batch_index]+X_HALF_WINDOW_PIXELS, location_y_samples[batch_index]-Y_HALF_WINDOW_PIXELS:location_y_samples[batch_index]+Y_HALF_WINDOW_PIXELS]
    # L Entry Channels (cannot vectorize here)
    for batch_index in range(batch_size):
        for line_index, line in enumerate(L_LINES):
            station_x_samples = (normalized_L_entries_compressed[line][day_index_samples[batch_index]][0] - location_x_samples[batch_index]-X_HALF_WINDOW_PIXELS).astype(np.int64)
            station_y_samples = (normalized_L_entries_compressed[line][day_index_samples[batch_index]][1] - location_y_samples[batch_index]-Y_HALF_WINDOW_PIXELS).astype(np.int64)
            entries = normalized_L_entries_compressed[line][day_index_samples[batch_index]][2]
            x_filter = (station_x_samples >= 0) & (station_x_samples < X_WINDOW_MAX_PIXELS)
            station_x_samples = station_x_samples[x_filter]
            station_y_samples = station_y_samples[x_filter]
            entries = entries[x_filter]
            y_filter = (station_y_samples >= 0) & (station_y_samples < Y_WINDOW_MAX_PIXELS)
            station_x_samples = station_x_samples[y_filter]
            station_y_samples = station_y_samples[y_filter]
            entries = entries[y_filter]
            mini_batch_input[batch_index, L_CHANNELS+line_index, station_x_samples, station_y_samples] = entries
    # Generate the corresponding output (Combine all categories for now)
    # Place the crimes on the map
    # - axis 0: day index
    # - axis 1: time slot
    # - axis 2: crime category
    # - axis 3: 0 is x locations, 1 is y locations, 2 is categories
    # - axis 4: crime locations
    # (cannot vectorize here)
    for batch_index in range(batch_size):
        for category_index in range(len(CRIME_CATEGORIES)):
            last_crime = np.argwhere(crime_frames[day_index_samples[batch_index]][time_slot_samples[batch_index]][category_index][0] == -1)[0][0]
            crimes_x = crime_frames[day_index_samples[batch_index]][time_slot_samples[batch_index]][category_index][0][:last_crime]
            crimes_y = crime_frames[day_index_samples[batch_index]][time_slot_samples[batch_index]][category_index][1][:last_crime]
            crimes_x = crimes_x - (location_x_samples[batch_index] - X_HALF_WINDOW_PIXELS)
            crimes_y = crimes_y - (location_y_samples[batch_index] - Y_HALF_WINDOW_PIXELS)
            x_filter = (crimes_x >= 0) & (crimes_x < X_WINDOW_MAX_PIXELS)
            crimes_x = crimes_x[x_filter]
            crimes_y = crimes_y[x_filter]
            y_filter = (crimes_y >= 0) & (crimes_y < Y_WINDOW_MAX_PIXELS)
            crimes_x = crimes_x[y_filter]
            crimes_y = crimes_y[y_filter]
            mini_batch_output[batch_index,crimes_x,crimes_y] = 1
    
    months_for_samples = month_fast_lookup[day_index_samples]
    one_hot_months = np.zeros((batch_size, 12))
    one_hot_months[np.arange(batch_size),months_for_samples] = 1
    
    days_for_samples = day_fast_lookup[day_index_samples]
    one_hot_days = np.zeros((batch_size, 31))
    one_hot_days[np.arange(batch_size),days_for_samples] = 1
    
    one_hot_time_slots = np.zeros((batch_size, NUM_TIME_SLOTS))
    one_hot_time_slots[np.arange(batch_size),time_slot_samples] = 1
    
    mini_batch_FC_inputs = np.concatenate([one_hot_months, one_hot_days, one_hot_time_slots, normalized_min_temp_lookup[day_index_samples,np.newaxis], normalized_max_temp_lookup[day_index_samples,np.newaxis], normalized_precipitation_lookup[day_index_samples,np.newaxis]], axis=1)
    
    return mini_batch_input, mini_batch_output, mini_batch_FC_inputs

EMPTY_SAMPLE = {'day_index' :np.array([]),
                'time_slot' :np.array([]),
                'location_x':np.array([]),
                'location_y':np.array([])}

def sample_index_and_location(num_samples, avoid_these_samples={'Test':EMPTY_SAMPLE,'Dev':EMPTY_SAMPLE}):
    # Randomly sample the time and location in our range.
    # Use this function primarily to avoid the test and dev sets.
    # avoid_these_samples is a dictionary with 'Test' and 'Dev' set indices as numpy arrays:
    # - day_index
    # - time_slot
    # - location_x
    # - location_y
    avoid_day_index = np.concatenate((avoid_these_samples['Test']['day_index'], avoid_these_samples['Dev']['day_index']))
    avoid_time_slot = np.concatenate((avoid_these_samples['Test']['time_slot'], avoid_these_samples['Dev']['time_slot']))
    avoid_location_x = np.concatenate((avoid_these_samples['Test']['location_x'], avoid_these_samples['Dev']['location_x']))
    avoid_location_y = np.concatenate((avoid_these_samples['Test']['location_y'], avoid_these_samples['Dev']['location_y']))
    # Create the empty sample arrays
    num_samples_taken = 0
    day_index_samples = np.zeros(num_samples, dtype=np.int64)
    time_slot_samples = np.zeros(num_samples, dtype=np.int64)
    x_location_samples = np.zeros(num_samples, dtype=np.int64)
    y_location_samples = np.zeros(num_samples, dtype=np.int64)
    # Generate the samples, avoiding the specified ones as necessary
    while (num_samples_taken < num_samples):
        day_index, time_slot, category, crime_index = choose_random_crime()
        location_x = crime_frames[day_index][time_slot][category][0][crime_index]
        location_y = crime_frames[day_index][time_slot][category][1][crime_index]
        # Need to randomize the location so that the crime is not in the center (watch out for the image boundaries)
        window_min_x = max(X_HALF_WINDOW_PIXELS, location_x - X_HALF_WINDOW_PIXELS+1)
        window_max_x = min(X_MAX_PIXELS-X_HALF_WINDOW_PIXELS, location_x + X_HALF_WINDOW_PIXELS-1)
        window_min_y = max(Y_HALF_WINDOW_PIXELS, location_y - Y_HALF_WINDOW_PIXELS+1)
        window_max_y = min(Y_MAX_PIXELS-Y_HALF_WINDOW_PIXELS, location_y + Y_HALF_WINDOW_PIXELS-1)
        location_x = (window_min_x + np.random.randint(window_max_x - window_min_x)) if window_max_x > window_min_x else window_min_x
        location_y = (window_min_y + np.random.randint(window_max_y - window_min_y)) if window_max_y > window_min_y else window_min_y
        # Only accept these if they do not overlap with samples we are avoiding
        day_should_be_avoided = np.isin(day_index_samples, day_index)
        time_slot_should_be_avoided = np.isin(time_slot_samples, time_slot)
        location_x_should_be_avoided = ((x_location_samples + X_HALF_WINDOW_PIXELS) > location_x) & ((x_location_samples - X_HALF_WINDOW_PIXELS) <= location_x)
        location_y_should_be_avoided = ((y_location_samples + Y_HALF_WINDOW_PIXELS) > location_y) & ((y_location_samples - Y_HALF_WINDOW_PIXELS) <= location_y)
        if np.any(day_should_be_avoided & time_slot_should_be_avoided & location_x_should_be_avoided & location_y_should_be_avoided):
            continue
        else:
            # No need to avoid this sample.  Add it to our test set.
            day_index_samples[num_samples_taken] = day_index
            time_slot_samples[num_samples_taken] = time_slot
            x_location_samples[num_samples_taken] = location_x
            y_location_samples[num_samples_taken] = location_y
            num_samples_taken+=1
    return {'day_index' : day_index_samples,
            'time_slot' :time_slot_samples,
            'location_x':x_location_samples,
            'location_y':y_location_samples}

def get_input_full_single(day_index, day, month, year, time_slot):
    input_data = np.zeros((NUM_INPUT_CHANNELS, X_MAX_PIXELS, Y_MAX_PIXELS))
    # Add static channels first
    input_data[:NUM_STATIC_CHANNELS] = static_channels
    # Weather channels
    input_data[MIN_TEMP_CHANNEL] = np.full((X_MAX_PIXELS, Y_MAX_PIXELS), min_temp_lookup[day_index])
    input_data[MAX_TEMP_CHANNEL] = np.full((X_MAX_PIXELS, Y_MAX_PIXELS), max_temp_lookup[day_index])
    input_data[PRECIPITATION_CHANNEL] = np.full((X_MAX_PIXELS, Y_MAX_PIXELS), precipitation_lookup[day_index])
    # Life Expectancy Channel
    input_data[LIFE_EXPECTANCY_CHANNEL] = life_expectancy_frame[year-FIRST_DATE.year]
    # L Entry Channels
    input_data[L_CHANNELS+0, L_entries_compressed['Green'][day_index][0], L_entries_compressed['Green'][day_index][1]] = L_entries_compressed['Green'][day_index][2]
    input_data[L_CHANNELS+1, L_entries_compressed['Red'][day_index][0], L_entries_compressed['Red'][day_index][1]] = L_entries_compressed['Red'][day_index][2]
    input_data[L_CHANNELS+2, L_entries_compressed['Brown'][day_index][0], L_entries_compressed['Brown'][day_index][1]] = L_entries_compressed['Brown'][day_index][2]
    input_data[L_CHANNELS+3, L_entries_compressed['Purple'][day_index][0], L_entries_compressed['Purple'][day_index][1]] = L_entries_compressed['Purple'][day_index][2]
    input_data[L_CHANNELS+4, L_entries_compressed['Yellow'][day_index][0], L_entries_compressed['Yellow'][day_index][1]] = L_entries_compressed['Yellow'][day_index][2]
    input_data[L_CHANNELS+5, L_entries_compressed['Blue'][day_index][0], L_entries_compressed['Blue'][day_index][1]] = L_entries_compressed['Blue'][day_index][2]
    input_data[L_CHANNELS+6, L_entries_compressed['Pink'][day_index][0], L_entries_compressed['Pink'][day_index][1]] = L_entries_compressed['Pink'][day_index][2]
    input_data[L_CHANNELS+7, L_entries_compressed['Orange'][day_index][0], L_entries_compressed['Orange'][day_index][1]] = L_entries_compressed['Orange'][day_index][2]
    # Date and Time channels
#     input_data[YEAR_CHANNEL + year - FIRST_DATE.year] = np.ones((X_MAX_PIXELS, Y_MAX_PIXELS))
#     input_data[MONTH_CHANNEL + month] = np.ones((X_MAX_PIXELS, Y_MAX_PIXELS))
#     input_data[DAY_CHANNEL + day] = np.ones((X_MAX_PIXELS, Y_MAX_PIXELS))
#     input_data[TIME_CHANNEL + time_slot] = np.ones((X_MAX_PIXELS, Y_MAX_PIXELS))
    return input_data

def get_expected_output_full_single(day_index, time_slot, category):
    # Create the map
    output_data = np.zeros((X_MAX_PIXELS, Y_MAX_PIXELS))
    # Place the crimes on the map
    # - axis 0: day index
    # - axis 1: time slot
    # - axis 2: crime category
    # - axis 3: 0 is x locations, 1 is y locations, 2 is categories
    # - axis 4: crime locations
    last_crime = np.argwhere(crime_frames[day_index][time_slot][category][0] == -1)[0][0]
    x_locations = crime_frames[day_index][time_slot][category][0][:last_crime]
    y_locations = crime_frames[day_index][time_slot][category][1][:last_crime]
    output_data[x_locations,y_locations] = 1
    return output_data

def calculate_mean_and_variance(test, dev, sample_size=1000):
    global INPUT_DATA_NORMALIZED
    INPUT_DATA_NORMALIZED = False
    # Generate a large number of inputs
    inputs, outputs, _ = generate_random_mini_batch(sample_size,
                                                    {'Test':test, 'Dev': dev},
                                                    layer_means=np.zeros(NUM_INPUT_CHANNELS),
                                                    layer_variances=np.ones(NUM_INPUT_CHANNELS),
                                                    normalize=False)
    # Calculate the mean and variance of each channel for normalizing all future mini-batches
    layer_means = np.zeros(NUM_INPUT_CHANNELS)
    layer_variances = np.ones(NUM_INPUT_CHANNELS)
    for layer in range(NUM_INPUT_CHANNELS):
        layer_means[layer] = np.mean(inputs[:,layer,:,:])
        layer_variances[layer] = np.var(inputs[:,layer,:,:])
    return layer_means, layer_variances

def transform_to_CNN_data(inputs, outputs, fc_inputs):
    m = inputs.shape[0]
    x = np.transpose(inputs, (0,2,3,1))
    y = np.sum(np.sum(outputs, axis=1), axis=1, dtype=np.int64)[:,np.newaxis]
    return x, y, fc_inputs

def create_placeholders():
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    input_height -- scalar, height of an input image
    input_width -- scalar, width of an input image
    input_channels -- scalar, number of channels of the input
    output_classes -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input
    Y -- placeholder for the input labels
    """

    X = tf.placeholder(tf.float32, shape=(None, input_image_height, input_image_width, NUM_INPUT_CHANNELS))
    X_FC = tf.placeholder(tf.float32, shape=(None, 12 + 31 + NUM_TIME_SLOTS + 3))
    Y = tf.placeholder(tf.float32, shape=(None, 1))
#     Y = tf.placeholder(tf.float32, shape=(None, num_anchor_boxes * output_grid_height * output_grid_width * 3 + 1))
    
    return X, Y, X_FC

def initialize_parameters():
    """
    Initializes weight parameters to build the CNN. The shapes are:
                        W1 : [4, 4, NUM_INPUT_CHANNELS, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
            
    W1 = tf.get_variable("W1", shape=(4,4,NUM_INPUT_CHANNELS,8), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", shape=(2,2,8,16), initializer=tf.contrib.layers.xavier_initializer(seed=0))

    W4 = tf.get_variable("W4", shape=(100, 64+(12+31+NUM_TIME_SLOTS+3)), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b4 = tf.get_variable("b4", shape=(100, 1), initializer=tf.zeros_initializer())
    W5 = tf.get_variable("W5", shape=(100, 100), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b5 = tf.get_variable("b5", shape=(100, 1), initializer=tf.zeros_initializer())
    W6 = tf.get_variable("W6", shape=(1, 100), initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b6 = tf.get_variable("b6", shape=(1, 1), initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "W2": W2,
                  "W4":W4,
                  "b4":b4,
                  "W5":W5,
                  "b5":b5,
                  "W6":W6,
                  "b6":b6}
    
    return parameters

def forward_propagation(X, X_FC, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1'] # CONV layer 1
    W2 = parameters['W2'] # CONV layer 2
    # layer 3 is a max-pool
    W4 = parameters['W4'] # FC layer 1
    b4 = parameters['b4']
    W5 = parameters['W5'] # FC layer 2
    b5 = parameters['b5']
    W6 = parameters['W6'] # FC layer 3
    b6 = parameters['b6']
    
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    A3 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    # FLATTEN
    A3 = tf.contrib.layers.flatten(A3)
    # Now combine the date layers
    A3 = tf.transpose(tf.concat([A3, X_FC], 1))
    Z4 = W4@A3+b4
    A4 = tf.nn.relu(Z4)
    Z5 = W5@A4+b5
    A5 = tf.nn.relu(Z5)
    # FULLY-CONNECTED without non-linear activation function
    Z6 = W6@Z5+b6
#     A6 = tf.nn.relu(Z6)

    return tf.transpose(Z6)

def compute_cost(Y_hat, Y):
    # Regression problem, so use mean squared error loss
#     cost = tf.reduce_mean((Y-Y_hat)**2)
    # Calculate the cost form the network prediction
    cost = tf.losses.mean_squared_error(labels=Y,
                                        predictions=Y_hat,
                                        reduction=tf.losses.Reduction.SUM)


    # NOTE: a relu is applied for regression.  For YOLO, we want partial relu, partial sigmoid
    # First output value is a regression:  We will use it as the K (number of crimes)
    # X, Y, Pc per anchor box per grid cell
    # num_crimes = tf.math.round(tf.nn.relu(Y_hat[0]))
    # cost_num_crimes = tf.reduce_mean((Y-num_crimes)**2)
    # anchor_grid = tf.reshape(tf.nn.sigmoid(Y_hat[1:]), (output_grid_width, output_grid_height, num_anchor_boxes, 3))
    # predicted_crime_locations = tf.where(anchor_grid[:,:,:,0] >= 0.5)
    # last_crime_index = tf.where(Y[:,0]==-1)[0]
    # crime_locations = Y[:,:last_crime_index]
    
    
    return cost

def compute_accuracy(Y_hat, Y):
    # Regression problem on number of crimes.
    # Round output to nearest, then compare.
    return tf.reduce_mean(tf.cast(tf.equal(tf.math.round(Y_hat), Y), "float"))

def restore_model(saver, session):
    # Before epoch, check for trial # in trial files
    if os.path.isfile(trial_file_location+trial_file_format % trial_number):
        print('Model found.  Restoring parameters.')
        # If trial exists:
        # Restore model
        saver.restore(session, pickled_model_location % trial_number)
    else:
        print('No saved model.  Using default parameter initialization.')

def epoch_teardown(saver, session, train_cost, dev_cost, training_accuracy, dev_accuracy, duration):
    trial_hyperparameters = pd.DataFrame(columns=hyperparameter_file_columns)
    # After epoch, check for hyperparameter file
    if os.path.isfile(trial_file_location+trial_file_format % trial_number):
        trial_hyperparameters = pd.read_excel(trial_file_location+trial_file_format % trial_number)
        # Save the model parameters
        saver.save(session, pickled_model_location % trial_number)
    # Save hyperparameters, epoch cost, and training & dev accuracies
    trial_hyperparameters = trial_hyperparameters.append({
        'Training Cost' : train_cost,
        'Dev Cost' : dev_cost,
        'Accuracy Evaluation Batch Size' : accuracy_eval_batch_size,
        'Maximum Crimes per Window' :max_num_crimes,
        'Number of Anchor Boxes per Window' : num_anchor_boxes,
        'Input Image Width' : input_image_width,
        'Input Image Height' : input_image_height,
        'Output Grid Width' : output_grid_width,
        'Output Grid Height' : output_grid_height,
        'Train Accuracy' : training_accuracy,
        'Dev Accuracy' : dev_accuracy,
        'Duration' : duration,
        'Learning Rate' : learning_rate,
        'Goal Total Epochs' : goal_total_epochs,
        'Minibatch Size' : minibatch_size,
        'Number of Minibatches' : num_minibatches,
        'Optimizer Name' : optimizer_name,
        'L2 Regularization Lambda' : regular_lambda,
        'Test Set Size' : test_set_size,
        'Dev Set Size' : dev_set_size,
        'Network Description' : network_description,
        'Samples for Normalization' : num_samples_for_normalization
    }, ignore_index=True)
    # Save the edited/new hyperparameter trial file
    writer = pd.ExcelWriter(trial_file_location+trial_file_format % trial_number)
    trial_hyperparameters.to_excel(writer)
    writer.save()

def execute_model():
    global optimizer_name
    
    # Generate the dev images for calculating dev accuracy during training
    x_dev, y_dev, x_fc_dev = transform_to_CNN_data(*generate_mini_batch(dev))

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    
    # Create Placeholders of the correct shape
    X, Y, X_FC = create_placeholders()
    # Initialize parameters
    parameters = initialize_parameters()
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Y_hat = forward_propagation(X, X_FC, parameters)
    
    # Cost and Accuracies
    cost = compute_cost(Y_hat, Y)
    accuracy = compute_accuracy(Y_hat, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    if optimizer_name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    else:
        optimizer_name = 'GD'
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        # Run the initialization
        session.run(init)
        # If the trial already exists, pick up where we left off
        restore_model(saver, session)
        # Do the training loop
        print('Beginning Training')
        x_train, y_train, x_fc_train = transform_to_CNN_data(*generate_random_mini_batch(accuracy_eval_batch_size, {'Test':test, 'Dev': dev}))
        for epoch in range(goal_total_epochs):
            start_time = time.time()
            epoch_cost = 0.
            for _ in range(num_minibatches):
                x_batch_train, y_batch_train, x_fc_batch_train = transform_to_CNN_data(*generate_random_mini_batch(minibatch_size, {'Test':test, 'Dev': dev}))
                _ , batch_cost = session.run([optimizer, cost], feed_dict={X: x_batch_train,
                                                                           Y: y_batch_train,
                                                                           X_FC: x_fc_batch_train})
                epoch_cost += batch_cost / num_minibatches
            elapsed_time = time.time() - start_time
            train_accuracy = accuracy.eval({X: x_train,
                                            Y: y_train,
                                            X_FC: x_fc_train})
            dev_accuracy = accuracy.eval({X: x_dev,
                                          Y: y_dev,
                                          X_FC: x_fc_dev})
            # Display epoch results every so often
            if epoch % epochs_between_prints == 0:
                print('%i Epochs' % epoch)
                print('\tCost: %f' % epoch_cost)
                print('\tTrain Accuracy: %f' % train_accuracy)
                print('\tDev Accuracy: %f' % dev_accuracy)
            # Epoch over, tear down
            dev_cost = cost.eval({X: x_dev, Y: y_dev, X_FC: x_fc_dev})
            epoch_teardown(saver,
                           session,
                           epoch_cost,
                           dev_cost,
                           float(train_accuracy),
                           float(dev_accuracy),
                           elapsed_time)                

        x_train, y_train, x_fc_train = transform_to_CNN_data(*generate_random_mini_batch(accuracy_eval_batch_size, {'Test':test, 'Dev': dev}))
        # Calculate the accuracy on the train and dev sets
        print('Reached Goal Number of Epochs.')
        print('Final Train Accuracy: %f' % accuracy.eval({X: x_train, Y: y_train, X_FC: x_fc_train}))
        print('Final Dev Accuracy: %f' % accuracy.eval({X: x_dev, Y: y_dev, X_FC: x_fc_train}))

##########################
# CREATE STATIC CHANNELS #
##########################

def create_static_channels():
    weather = None
    if not USING_EC2:
        weather = pd.read_excel(processed_dataset_paths_xlsx % 'Weather')
    else:
        weather = pd.read_excel(s3.get_object(Bucket= bucket, Key= 'CNN Input (Sean)/Weather.xlsx') ['Body'])
    weather['Date'] = pd.to_datetime(weather['Date'])
    # Create fast-access weather arrays
    min_temp_lookup = np.full((NUM_DAYS), np.nan)
    max_temp_lookup = np.full((NUM_DAYS), np.nan)
    precipitation_lookup = np.full((NUM_DAYS), np.nan)
    # Insert the weather data
    weather.apply(lambda record: extract_data_for_date(record, max_temp_lookup, 'Max Temp'), axis=1)
    weather.apply(lambda record: extract_data_for_date(record, min_temp_lookup, 'Min Temp'), axis=1)
    weather.apply(lambda record: extract_data_for_date(record, precipitation_lookup, 'Precipitation'), axis=1)
    # Interpolate over any NaN values
    nans, x= nan_helper(min_temp_lookup)
    min_temp_lookup[nans]= np.interp(x(nans), x(~nans), min_temp_lookup[~nans])
    nans, x= nan_helper(max_temp_lookup)
    max_temp_lookup[nans]= np.interp(x(nans), x(~nans), max_temp_lookup[~nans])
    nans, x= nan_helper(precipitation_lookup)
    precipitation_lookup[nans]= np.interp(x(nans), x(~nans), precipitation_lookup[~nans])

    # Load static data
    street_frame = None
    waterway_frame = None
    park_frame = None
    forest_frame = None
    school_frame = None
    library_frame = None
    uninhabitable_building_frame = None
    building_frames = {'Sound':{},
                       'Minor Repair':{},
                       'Major Repair':{}}
    life_expectancy_frame = None
    business_frames = {}
    L_entries_compressed = None
    socioeconomic_frames = None
    if not USING_EC2:
        street_frame = np.load(dataset_location + 'Streets Frame.npz')['street_frame']
        waterway_frame = np.load(dataset_location + 'Waterway Frame.npz')['waterway_frame']
        park_frame = np.load(dataset_location + 'Park Frame.npz')['park_frame']
        forest_frame = np.load(dataset_location + 'Forest Frame.npz')['forest_frame']
        school_frame = np.load(dataset_location + 'School Frame.npz')['school_frame']
        library_frame = np.load(dataset_location + 'Library Frame.npz')['library_frame']
        uninhabitable_building_frame = np.load(dataset_location + 'Building Frames.npz')['uninhabitable_building_frame']
        with np.load(dataset_location + 'Building Frames.npz') as data:
            building_frames['Sound']['Stories'] = data['stories_of_sound_buildings_frame']
            building_frames['Sound']['Area'] = data['area_of_sound_buildings_frame']
            building_frames['Sound']['Units'] = data['units_of_sound_buildings_frame']

            building_frames['Minor Repair']['Stories'] = data['stories_of_minor_repair_buildings_frame']
            building_frames['Minor Repair']['Area'] = data['area_of_minor_repair_buildings_frame']
            building_frames['Minor Repair']['Units'] = data['units_of_minor_repair_buildings_frame']

            building_frames['Major Repair']['Stories'] = data['stories_of_major_repair_buildings_frame']
            building_frames['Major Repair']['Area'] = data['area_of_major_repair_buildings_frame']
            building_frames['Major Repair']['Units'] = data['units_of_major_repair_buildings_frame']
        life_expectancy_frame = np.load(dataset_location + 'Life Expectancy Frames.npz')['life_expectancy_frame']
        with np.load(dataset_location + 'Business Frames.npz') as data:
            business_frames['Food Service'] = data['Food Service']
            business_frames['Tobacco Sale'] = data['Tobacco Sale']
            business_frames['Alcohol Consumption'] = data['Alcohol Consumption']
            business_frames['Package Store'] = data['Package Store']
            business_frames['Gas Station'] = data['Gas Station']
        L_entries_compressed = pd.read_csv(dataset_location + 'L Entries.csv')
        L_entries_compressed_as_array = np.zeros((len(L_entries_compressed), len(L_LINES)))
        # Unpack the json strings to numpy
        for line in L_LINES:
            L_entries_compressed[line] = L_entries_compressed[line].apply(lambda array_string: np.array(json.loads(array_string)))
        # L Entries is a pandas dataframe:
        #  column is L line
        #  row is day number
        #  Cell is numpy array:
        #    row 1 is x coordinate of rail station
        #    row 2 is y coordinate of rail station
        #    row 3 is number of entries for rail station
        socioeconomic_frames = np.load(dataset_location + 'Socioeconomic Frames.npz')['socioeconomic_frame']
    else:
        street_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Streets Frame.npz')['Body'].read()))['street_frame']
        waterway_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Waterway Frame.npz')['Body'].read()))['waterway_frame']
        park_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Park Frame.npz')['Body'].read()))['park_frame']
        forest_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Forest Frame.npz')['Body'].read()))['forest_frame']
        school_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/School Frame.npz')['Body'].read()))['school_frame']
        library_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Library Frame.npz')['Body'].read()))['library_frame']
        uninhabitable_building_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Building Frames.npz')['Body'].read()))['uninhabitable_building_frame']
        with np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Building Frames.npz')['Body'].read())) as data:
            building_frames['Sound']['Stories'] = data['stories_of_sound_buildings_frame']
            building_frames['Sound']['Area'] = data['area_of_sound_buildings_frame']
            building_frames['Sound']['Units'] = data['units_of_sound_buildings_frame']

            building_frames['Minor Repair']['Stories'] = data['stories_of_minor_repair_buildings_frame']
            building_frames['Minor Repair']['Area'] = data['area_of_minor_repair_buildings_frame']
            building_frames['Minor Repair']['Units'] = data['units_of_minor_repair_buildings_frame']

            building_frames['Major Repair']['Stories'] = data['stories_of_major_repair_buildings_frame']
            building_frames['Major Repair']['Area'] = data['area_of_major_repair_buildings_frame']
            building_frames['Major Repair']['Units'] = data['units_of_major_repair_buildings_frame']
        life_expectancy_frame = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Life Expectancy Frames.npz')['Body'].read()))['life_expectancy_frame']
        with np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Business Frames.npz')['Body'].read())) as data:
            business_frames['Food Service'] = data['Food Service']
            business_frames['Tobacco Sale'] = data['Tobacco Sale']
            business_frames['Alcohol Consumption'] = data['Alcohol Consumption']
            business_frames['Package Store'] = data['Package Store']
            business_frames['Gas Station'] = data['Gas Station']
        L_entries_compressed = pd.read_csv(s3.get_object(Bucket= bucket, Key= 'CNN Input (Sean)/L Entries.csv') ['Body'])
        L_entries_compressed_as_array = np.zeros((len(L_entries_compressed), len(L_LINES)))
        # Unpack the json strings to numpy
        for line in L_LINES:
            L_entries_compressed[line] = L_entries_compressed[line].apply(lambda array_string: np.array(json.loads(array_string)))
        # L Entries is a pandas dataframe:
        #  column is L line
        #  row is day number
        #  Cell is numpy array:
        #    row 1 is x coordinate of rail station
        #    row 2 is y coordinate of rail station
        #    row 3 is number of entries for rail station
        socioeconomic_frames = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Socioeconomic Frames.npz')['Body'].read()))['socioeconomic_frame']
    
    static_channels = np.zeros((NUM_STATIC_CHANNELS, X_MAX_PIXELS, Y_MAX_PIXELS))
    static_channels[STREET_CHANNEL] = street_frame
    static_channels[WATERWAY_CHANNEL] = waterway_frame
    static_channels[PARK_CHANNEL] = park_frame
    static_channels[FOREST_CHANNEL] = forest_frame
    static_channels[SCHOOL_CHANNEL] = school_frame
    static_channels[LIBRARY_CHANNEL] = library_frame
    static_channels[BUILDING_CHANNELS + 0] = uninhabitable_building_frame
    static_channels[BUILDING_CHANNELS + 1] = building_frames['Sound']['Stories']
    static_channels[BUILDING_CHANNELS + 2] = building_frames['Sound']['Area']
    static_channels[BUILDING_CHANNELS + 3] = building_frames['Sound']['Units']
    static_channels[BUILDING_CHANNELS + 4] = building_frames['Minor Repair']['Stories']
    static_channels[BUILDING_CHANNELS + 5] = building_frames['Minor Repair']['Area']
    static_channels[BUILDING_CHANNELS + 6] = building_frames['Minor Repair']['Units']
    static_channels[BUILDING_CHANNELS + 7] = building_frames['Major Repair']['Stories']
    static_channels[BUILDING_CHANNELS + 8] = building_frames['Major Repair']['Area']
    static_channels[BUILDING_CHANNELS + 9] = building_frames['Major Repair']['Units']
    static_channels[BUSINESS_CHANNELS + 0] = business_frames['Food Service']
    static_channels[BUSINESS_CHANNELS + 1] = business_frames['Tobacco Sale']
    static_channels[BUSINESS_CHANNELS + 2] = business_frames['Alcohol Consumption']
    static_channels[BUSINESS_CHANNELS + 3] = business_frames['Package Store']
    static_channels[BUSINESS_CHANNELS + 4] = business_frames['Gas Station']
    static_channels[SOCIO_CHANNELS + 0] = socioeconomic_frames[0]
    static_channels[SOCIO_CHANNELS + 1] = socioeconomic_frames[1]
    static_channels[SOCIO_CHANNELS + 2] = socioeconomic_frames[2]
    static_channels[SOCIO_CHANNELS + 3] = socioeconomic_frames[3]
    static_channels[SOCIO_CHANNELS + 4] = socioeconomic_frames[4]
    static_channels[SOCIO_CHANNELS + 5] = socioeconomic_frames[5]
    static_channels[SOCIO_CHANNELS + 6] = socioeconomic_frames[6]

    return static_channels, min_temp_lookup, max_temp_lookup, precipitation_lookup, life_expectancy_frame, L_entries_compressed

static_channels, min_temp_lookup, max_temp_lookup, precipitation_lookup, life_expectancy_frame, L_entries_compressed = create_static_channels()
# Generate global normalized data arrays
normalized_zeros = np.zeros(NUM_INPUT_CHANNELS)
# Normalize the static channels
normalized_static_channels = static_channels

# Normalize the weather data
normalized_min_temp_lookup = min_temp_lookup
normalized_max_temp_lookup = max_temp_lookup
normalized_precipitation_lookup = precipitation_lookup
# Normalize life expectancy
normalized_life_expectancy_frame = life_expectancy_frame
# Normalize L Entries
normalized_L_entries_compressed = L_entries_compressed
# Calculate normalized date and time 'ones'
# normalized_months = np.ones(12)
# normalized_days = np.ones(31)
# normalized_time_slots = np.ones(NUM_TIME_SLOTS)
INPUT_DATA_NORMALIZED = False
print('Input data loaded.')

#####################################
# IMPORT PROCESSED (NONSTATIC) DATA #
#####################################

crime_frames = None
if not USING_EC2:
    crime_frames = np.load(dataset_location + 'Crimes.npz')['crime_frame']
else:
    crime_frames = np.load(BytesIO(s3.get_object(Bucket=bucket, Key='CNN Input (Sean)/Crimes.npz')['Body'].read()))['crime_frame']
# Make it easy to convert from day_index to year, month, day
year_fast_lookup = np.vectorize(lambda day_index: (FIRST_DATE+dtm.timedelta(days=int(day_index))).year)(np.arange(NUM_DAYS))
month_fast_lookup = np.vectorize(lambda day_index: (FIRST_DATE+dtm.timedelta(days=int(day_index))).month)(np.arange(NUM_DAYS))-1
day_fast_lookup = np.vectorize(lambda day_index: (FIRST_DATE+dtm.timedelta(days=int(day_index))).day)(np.arange(NUM_DAYS))-1
# Make it easy to randomly choose crimes
crime_indices = np.argwhere(crime_frames != -1)
print('Output data loaded.')

# Create the Test and Dev sets
test = sample_index_and_location(test_set_size)
dev = sample_index_and_location(dev_set_size, {'Test':test, 'Dev':EMPTY_SAMPLE})
print('Test and Dev samples generated.')
# Ensure the data is not normalized
normalize_input_data(np.zeros(NUM_INPUT_CHANNELS), np.ones(NUM_INPUT_CHANNELS))
layer_means, layer_variances = calculate_mean_and_variance(test, dev, sample_size=num_samples_for_normalization)
print('Mean and variance calculated.')
normalize_input_data(layer_means, layer_variances)
print('Input data normalized')

execute_model()