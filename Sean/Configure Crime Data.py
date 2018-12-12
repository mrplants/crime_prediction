import numpy as np
import pandas as pd
import math
import calendar
import boto3
from io import StringIO

#################
# S3 PARAMETERS #
#################
def s3_read_excel(objname):
    bucket = "cs230"
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket= bucket, Key= file_name)
    return pd.read_excel(obj['Body'])
def s3_read_csv(objname):
    bucket = "cs230"
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket= bucket, Key= objname)
    return pd.read_csv(obj['Body'])

###################
# LOAD CRIME DATA #
###################
# crime_dataframe = pd.read_csv('/Volumes/GoogleDrive/My Drive/Crime Data/Raw Data/Crimes (Chicago).csv')
crime_dataframe = s3_read_csv('Crimes (Chicago).csv')
print('Crime data loaded.')

########################
# CONDITION CRIME DATA #
########################
# Delete columns that are redundant or unhelpful
columns_to_delete = ['Case Number', 'Location Description', 'Block', 'Arrest', 'Domestic', 'FBI Code', 'Primary Type', 'Description','X Coordinate', 'Y Coordinate', 'Year', 'Updated On', 'Location']
final_dataframe = crime_dataframe.drop(columns=columns_to_delete).copy().dropna()
print('Unnecessary crime columns removed.')
print('\tLength: %d' % len(final_dataframe))

##################################
# CONVERT CRIME STATS TO ONE-HOT #
##################################
# Convert 'Beat', 'District', 'Ward', and 'Community Area' to one-hot vectors
beat_one_hot = pd.get_dummies(final_dataframe['Beat'].astype(str).apply(lambda x: 'BEAT_'+x))
district_one_hot = pd.get_dummies(final_dataframe['District'].astype(str).apply(lambda x: 'DISTRICT_'+x))
ward_one_hot = pd.get_dummies(final_dataframe['Ward'].astype(str).apply(lambda x: 'WARD_'+x))
community_one_hot = pd.get_dummies(final_dataframe['Community Area'].astype(str).apply(lambda x: 'COMMUNITY_'+x))
final_dataframe = pd.concat([final_dataframe.drop(columns=['Beat', 'District', 'Ward', 'Community Area']),
                            beat_one_hot,
                            district_one_hot,
                            ward_one_hot,
                            community_one_hot], axis=1).dropna()
print('Beat, District, Ward, and Community Area converted to one-hot vectors.')
print('\tLength: %d' % len(final_dataframe))

##################
# CONDITION DATE #
##################
# Convert crime dates to YEAR, MONTH, DAY, HOUR, MINUTE, and weekday columns
# Convert those to one-hot and concat with final dataframe
final_dataframe['Date'] = pd.to_datetime(crime_dataframe['Date'])
year_one_hot = pd.get_dummies(final_dataframe['Date'].dt.year.astype(str).apply(lambda x: 'YEAR_'+x))
month_one_hot = pd.get_dummies(final_dataframe['Date'].dt.month.apply(lambda x: calendar.month_abbr[x]))
day_one_hot = pd.get_dummies(final_dataframe['Date'].dt.day.astype(str).apply(lambda x: 'DAY_'+x))
hour_one_hot = pd.get_dummies(final_dataframe['Date'].dt.hour.astype(str).apply(lambda x: 'HOUR_'+x))
minute_one_hot = pd.get_dummies(final_dataframe['Date'].dt.minute.astype(str).apply(lambda x: 'MINUTE-'+x))
weekday_one_hot = pd.get_dummies(final_dataframe['Date'].dt.weekday.apply(lambda x: calendar.day_name[x]))
final_dataframe = pd.concat([final_dataframe,
                            year_one_hot,
                            month_one_hot,
                            day_one_hot,
                            hour_one_hot,
                            minute_one_hot,
                            weekday_one_hot], axis=1).dropna()
print('Date, time, and day converted to one-hot vectors.')
print('\tLength: %d' % len(final_dataframe))

####################
# JOIN TEMPERATURE #
####################
# temperature_dataframe = pd.read_csv('/Volumes/GoogleDrive/My Drive/Crime Data/Raw Data/Temperatures (Chicago).csv')
temperature_dataframe = s3_read_csv('Temperatures (Chicago).csv')
# Drop the TAVG column because it has too many NaNs
temperature_dataframe = temperature_dataframe.drop(columns=['TAVG'])
# Convert the Precipitation, max T, and min T columns to float
temperature_dataframe['PRCP'] = pd.to_numeric(temperature_dataframe['PRCP'])
temperature_dataframe['TMAX'] = pd.to_numeric(temperature_dataframe['TMAX'])
temperature_dataframe['TMIN'] = pd.to_numeric(temperature_dataframe['TMIN'])
temperature_dataframe.rename(columns={'PRCP':'PRECIPITATION'})
# Join with the final dataframe
temperature_dataframe['DATE'] = pd.to_datetime(temperature_dataframe['DATE'])
temperature_dataframe['DAY'] = temperature_dataframe['DATE'].dt.day
temperature_dataframe['MONTH'] = temperature_dataframe['DATE'].dt.month
temperature_dataframe['YEAR'] = temperature_dataframe['DATE'].dt.year
final_dataframe['DAY'] = final_dataframe['Date'].dt.day
final_dataframe['MONTH'] = final_dataframe['Date'].dt.month
final_dataframe['YEAR'] = final_dataframe['Date'].dt.year
final_dataframe = final_dataframe.merge(temperature_dataframe, on=['DAY', 'MONTH', 'YEAR'], how='left').drop(columns=['DATE', 'STATION', 'NAME', 'DAY', 'MONTH', 'YEAR']).dropna()
print('Temperature and precipitation merged into final dataset.')
print('\tLength: %d' % len(final_dataframe))

# Write the new crime data to a temporary file in my workspace
# For Local Machine
# writer = pd.ExcelWriter('/Volumes/GoogleDrive/My Drive/Crime Data/Composite Data/Sean Workspace/23_November.xlsx')
# final_dataframe.to_excel(writer)
# writer.save()
# For AWS
write_buffer = StringIO()
final_dataframe.to_excel(write_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object('cs230', '25_November.xlsx').put(Body=csv_buffer.getvalue())
