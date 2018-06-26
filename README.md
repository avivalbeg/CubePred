# Read Me Doc 
This is the Read Me Doc for RasterClass and MemClass which is made for the OpenDataCube project by **Aviv Elbag**,  **Otto Wagner**, and **AMA**. 

Contact: **aviv.m.elbag@ama-inc.com** for questions.


# Goal
The goal of these two modules (rasterclass and memClass) are to 
make a basis for a general classification base using geospatial
data in the datacube. The memClass and the rasrerClass modules 
allow for the use, development, and storage framework of
classification schemes of geospatial extents. 

It's goal is to be powerful, flexible, and simple to use. For a 
quick way to get started see examples: 

- **Condensed RasterClass.ipynb**
- **Condensed memClass.ipynb**



## RasterClass
For source see MemClass.py

### Goal

RasterClass is a class to handle shape files. Shape files can be 
cumbersome when dealing with raster data such as that found in 
landsat data. 


### Dependencies
 - **rasterio**    
     For the affine transform method
     
 - **geopandas**
     an easy to use pandas interface for geospatial applications
     
 - **pyproj**
     for coordinate reference systems parsing
     
 - **numpy**
     ...It's numpy. It might as well be included inside python core.
     
 - **shapely.geometry**
     for taking in shape files
     
 - **scipy.misc**
     for resampling images
     
 - **matplotlib.pyplot**
     for plotting
 
 - **os**
     Is it used... I don't think so...
     


### Possible Issues


### List of methods

#### binary_rasterize
    produces a binary raster in a bounding box of a given shape file of boolean values to describe whether or not the shape includes a pixel.
    Inputs:
        Data(geopandas.DataFrame): A list of shape files or a single shape file
        data_label: unused
        data_value: If Data is a list of shape files with labels then fata_value will look for a particular label
        prev_Map: (optional)An array which you want to be overlayed
        size:(optional) size of the desired output map in pixels. Else it will automatically find the size necessary.
    
    Output: 
        new_Map(numpy array): A boolean array of values that correspond to the bounding box of the shape file.(This is in the coordinates of the designated CRS)
        minx(float): the minimum X bound
        maxx(float): the maximum X bound
        miny(float): the minimum Y bound
        maxy(float): the maximum Y bound

    Example: TODO
    

#### clip
    Takes a raster of a map and a raster of the binary_shape raster(not necessarily resized) and outputs a masked map.
    
    Input:
        truth(numpy array): The bounded map of interest
        img2(numpy array): The boolean binary_shape raster
    Output:
        imga(numpy array): The resampled binary raster
        imgf(numpy array): The clipped map.(Same resolution as truth)
    
    Example: TODO
        
        

#### clip_from_shape
    Same as clip but instead of inputting the rastered shape you just input the raw shape. 
    Input:
        shape(shapely.geometry): Take a shape and outputs a bounded clip of the area. 
    Output:
        img(numpy array): binary_shape raster
        img1(numpy array): the resized binary_shape raster(imga)
        img2(numpy array): the bounded map(unclipped)
        imgClipped(numpy array): the clipped map (imgf from clip)
        
    Example: TODO
        

#### composite
    To remove clouds and obstructions in a series of images of the same extent.
    Input:
     ds(xarray): A datacube dataset with pixel_qa as one of it's measurements
    
    Output:
     An image with median values for the pixels and all possible obstructions removed.

#### get_map
    Gets a map from datacube.
    
    Input:
        xmin(float): The minimum x coordinate in the CRS defined.
        xmax(float): The maximum x coordinate in the CRS defined.
        ymin(float): The minimum y coordinate in the CRS defined.
        ymax(float): The maximum y coordinate in the CRS defined.
        product(string): The datacube product of interest(To see available products use your defined datacube object(dc) and call dc.list_products())
        coordinate_system(string): The coordinate system usually in this format: 'epsg:4326' default is epsg:4326
        time(tuple): A tuple of bounding dates. i.e '('2012-01-01', '2015-01-01')'
        measurements(tuple): A list of measurements you want from the product. i.e. '['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'pixel_qa']'
        
    Output:
        img(numpy array): A composited raster(between the dates defined) of the area defined with undefined areas(SLC bands and perpetual clouds) set to zero(black).      

#### is_l7_dataset_clean
     Only shows areas in an image that are clear from clouds or otherwise are visibile to the gorund.
    
    Input:
    dataset(xarray): A datacube dataset
    
    Output: 
    a_clean_mask(numpy array): A numpy array of booleans showing where the image data is good.

#### rasterize_and_separate
    Takes several classified shapely shapes and makes a colored map.(Not usually used)
    Input:
        Geometries(shapely shapes): Array of shapes
        Transform(Affine Transform): 
            Optional - example: T = transform.from_origin(634000,242000,1,1) 
        prev_Map(numpy array): Optional
        
    Output:
        prev_Map(numpy array): An array of values that correspond to categorizations of the raster.
     
    Example: TODO
        

#### read_folder
    Reads a folder for all the shape files.
    Input:
        None
    Output:
        one(geopandas dataframe): A list of all the shapes
        
    Example: TODO


#### run_example
    An example to show capabilities
    
    Example: TODO


### Future Work

    Make it cleaner. Also, convert to different formats. Possibly use sparse arrays for more efficiency. 
    
    TODO: MORE LATER.


## MemClass

 - For source see MemClass.py
 
### Goal

MemClass is made as a framework for a general classification. The idea is that this is a way to easily collect and create classifications that can then be compared in a robust and simple way. 

### Dependencies
 - **RasterClass**    
     For converting shape files to raster files
     
 - **sklearn.ensemble**
     sklearn Random Forests are included here
     
 - **sklearn.cluster**
     sklearn Kmeans is included here
     
 - **numpy**
     ...It's numpy. It might as well be included inside python core.
     
 - **matplotlib.pyplot**
     for plotting
     
 - **os**
     for loading and saving
     
 - **pickle**
     for loading and saving


### Possible Issues

MemClass uses datacube which is still liable to change. 


### List of methods


#### Columbia_Forest_Predictor(Yes I know it's spelled wrong)
    Input: 
        mode - Tells the predictor how it should behave
            1. auto  ---> Does default behavior of showing the sample classifier
            2. load  ---> takes an input classifier. Requires filename keyword defined to be the location of the classifier.
            3. shape_folder ---> takes an input of a folder that includes shape files. Requires filename keyword defined to be the location of the directory of interest. This will initiate a user interface.
       
        filename - Used when the mode is 'load' or 'shape_folder'.
            - when mode = 'load' it should point to the pickled classifier of interest
            - when mode = 'shape_folder' it should point to a directory with the shape files of interest.
        
    Output: An instance of the Columbia_Forest_Predictor this is actually a bad name as the predictor is general. Once you have the classifer in an instance you can use the following methods from it. 
      
##### protected read_folder
    Input:

#####  protected read_shapes

##### public binary_rasterize

##### clip

##### clip_from_shape

##### coTran

##### composite

##### get_map

##### imresize

##### is_l7_dataset_clean

##### rasterize

##### rasterize_and_separate

##### read_folder

##### run_example


### Future Work







