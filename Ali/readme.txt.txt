

## Explanatin of the folders

best_results/ 

    Includes the files that have the best models (cpkt files)


cnn_data_engineering_related_old/

    Includes the old ways of generating data. There were 2:

        1) Building a web application and taking screenshots of Google Map APIs

        2) Taking a single screenshot of Google Map APIs and then getting latitude/longitude ratios and manually drawing dots 
        onto the base image. 

cnn_data_engineering_related_new/

        Using Shapely python library, we converted each latitude/longitude value to a pixel value on 256x256 image. This worked the best
    for using the GIS datasets available online. 

notebook_models/

    These are the training iterations of the model taken

notebook_baseline_simple_models/

    These are the simple algorithms we used to baseline model


Remember, raw data to be fed into the model is located at: https://drive.google.com/drive/folders/1LOyrqORS4Zszm4n62siat888IIgpx4rx?usp=sharing
