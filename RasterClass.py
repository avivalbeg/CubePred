
from rasterio import transform
import geopandas as gpd
from pyproj import Proj
from pyproj import transform as coTran
import datacube
import numpy as np
from shapely.geometry import shape
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from scipy.misc import imresize
import os

# T = transform.from_origin(634000,242000,1,1)
b = shape({
    'type': 'Polygon',
'coordinates': [[(2, 2), (2, 90), (90, 90), (4.25, 2), (2, 2)]]})
dc = datacube.Datacube(app    = "my_random_app",
                       config = "/home/localuser/.datacube.conf")

def is_l7_dataset_clean(dataset):
    """
    Only shows areas in an image that are clear from clouds or otherwise are visibile to the gorund.
    
    Input:
    dataset(xarray): A datacube dataset
    
    Output: 
    a_clean_mask(numpy array): A numpy array of booleans showing where the image data is good.
    """
    print(dataset)
    clear_pixels = dataset.pixel_qa.values == 2 + 64 #clear(no obstructions of view) pixels is where pixel_qa is 66 
    water_pixels = dataset.pixel_qa.values == 4 + 64 #water pixels is where pixel_qa is 68
    a_clean_mask = np.logical_or(clear_pixels, water_pixels)   #returns a mask of only pixels that are usable
    return a_clean_mask

def composite(ds):
    """
    To remove clouds and obstructions in a series of images of the same extent.
    Input:
     ds(xarray): A datacube dataset with pixel_qa as one of it's measurements
    
    Output:
     An image with median values for the pixels and all possible obstructions removed.
    
    """
    ds = ds.where(is_l7_dataset_clean(ds))
    return ds.median(dim='time', skipna=True, keep_attrs=False)


def rasterize_and_separate(Geometries,transform= None, prev_Map = None,data_label = []):
    if prev_Map == None:
        prev_Map = np.zeros((3000,2000))
    for i,geom in enumerate(Geometries.geometry):
        if len(data_label) == 0:
            new_Map = rasterize([geom], out_shape = prev_Map.shape,default_value = i+1,transform=transform)
        else:
            new_Map = rasterize([geom], out_shape = prev_Map.shape,default_value = data_label.values[i],transform=transform)
            
        prev_Map+=new_Map
    return prev_Map

def binary_rasterize(Data,
                     data_label=None,
                     data_value=None,
                     prev_Map = None,
                     size= None):
    """
    This function has two modes. It has one mode where you just input the geometry and will 
    rasterize the data into true and false. In the bounding box of the geometry. 
    The other mode takes a Dataframe a data label a data value and returns all geometries in
    raster form that were of that data value.
    Geometries:
    data_label:
    data_value:
    transform:
    prev_Map:
    size:
    
    Output:
    """
    if prev_Map == None: #Even if size is defined we don't want to use it if the map is defined
        if type(Data) is not type(b):  # could be optimized
            MinX,MinY,MaxX,MaxY = (min(Data.geometry.bounds['minx']),min(Data.geometry.bounds['miny'])
                                   ,max(Data.geometry.bounds['maxx']),max(Data.geometry.bounds['maxy']))
        else:
            MinX,MinY,MaxX,MaxY = Data.bounds
        if size == None:     
            prev_Map = np.zeros((int(MaxY-MinY),int(MaxX-MinX)))
        else:
            prev_Map = np.zeros(size)
        
    else:
        prev_Map = prev_Map
        
    if type(Data) != type(b):
        DataGeom= Data[Data[data_label]==data_value]
        for geom in DataGeom.geometry:
            minx,miny,maxx,maxy = geom.bounds
            T = transform.from_origin(minx,maxy,1,1)
            new_Map = rasterize([geom], out_shape = prev_Map.shape
                                ,default_value = 1,transform=T)
            prev_Map+=new_Map
        try:
            print('good')
            Minx,Miny = np.where( new_Map==np.min(new_Map[np.nonzero(new_Map)]))
            Maxx,Maxy = np.where( new_Map==np.max(new_Map[np.nonzero(new_Map)]))
            Minx= min(Minx)
            Maxx = max(Maxx)
            Miny= min(Miny)
            Maxy = max(Maxy)
            new_Map=new_Map[Minx:Maxx,Miny:Maxy]
            return new_Map,minx,miny,maxx,maxy
        except:
            print("It seems that your label doesn't have any elements... Or there is a bug. ")
    
    else:
        minx,miny,maxx,maxy = Data.bounds
        T = transform.from_origin(minx,maxy,1,1)
        new_Map = rasterize([Data], out_shape = prev_Map.shape
                            ,default_value = 1,transform=T)
        prev_Map+=new_Map
        return new_Map,minx,miny,maxx,maxy

def get_map(xmin,
            xmax,
            ymin,
            ymax,
            product = "ls7_ledaps_sanagustin",
            coordinate_system = 'epsg:4326',time = None,
            measurements = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'pixel_qa']):
    
    dc = datacube.Datacube(app    = "my_random_app",
                       config = "/home/localuser/.datacube.conf")

#     xmin = coordinates[0]
#     xmax = coordinates[0]+size[0]
#     ymin = coordinates[1]
#     ymax = coordinates[1]+size[1]
    
    
#     projFrom = Proj(init = coordinate_system)     #The coordinate system used in the data
#     projTo = Proj(init = 'epsg:4326')    #WGS84

#     lonmin,latmin= coTran(projFrom,projTo,xmin,ymin)   #transform from epsg:... to WGS84

#     lonmax,latmax= coTran(projFrom,projTo,xmax,ymax)

#     latmean = np.average([latmin,latmax])
#     lonmean = np.average([lonmin,lonmax])
    dataset = dc.load( product = product,x = (xmin, xmax),y = (ymin, ymax),crs = coordinate_system,measurements = measurements,  time = ('2012-01-01', '2015-01-01')) 


    median_dataset = composite(dataset)
    img = np.dstack((median_dataset.swir1.values/np.nanmax(median_dataset.swir1.values),median_dataset.nir.values/np.nanmax(median_dataset.nir.values),median_dataset.red.values/np.nanmax(median_dataset.red.values)))
    img = np.nan_to_num(img,0)
    return img
    

def clip(truth,img2):
    imga= imresize(truth,img2.shape)
    imgf = np.einsum('ijk,ij-> ijk',img2,imga)
    return imga,imgf
          
def clip_from_shape(shape):
    img,minx1,maxx1,miny1,maxy1= binary_rasterize(shape)
    img2 = get_map(minx1,miny1,maxx1,maxy1,coordinate_system='EPSG:32618')
    img1,imgClipped= clip(img,img2)
    return (img,img1,img2,imgClipped)

def read_folder():
    one = gpd.GeoDataFrame()
    for file in os.listdir():
        if file.endswith(".shp"):
            One = gpd.read_file(file)
            one = one.append(One)
    return one



def run_example():
    #plt.imshow(get_map((643000,200000),coordinate_system='EPSG:32618'))  
    #T = transform.from_origin(634000,242000,1,1)
    one = read_folder()
    for elem in one.geometry:
        A = elem
        break    	
    #img,minx1,maxx1,miny1,maxy1= binary_rasterize(A)
    #plt.imshow(img)
    #plt.show()
    # img2 = get_map((minx1,miny1),size = (maxx1-minx1,maxy1-miny1),coordinate_system='EPSG:32618')
    #img2 = get_map(minx1,miny1,maxx1,maxy1,coordinate_system='EPSG:32618')
    #plt.imshow(img2)
    #plt.show()
  #     if prev_Map == None:
	#         try:
	#             prev_Map = n 
    #plt.imshow(rasterize_and_separate(A,transform = T))
    #plt.imshow(A)
    [img,img1,img2,imgClipped] = clip_from_shape(A)
    plt.imshow(img)
    plt.show()
    plt.imshow(img1)
    plt.show()
    # img2 = get_map((minx1,miny1),size = (maxx1-minx1,maxy1-miny1),coordinate_system='EPSG:32618')
    #img2 = get_map(minx1,miny1,maxx1,maxy1,coordinate_system='EPSG:32618')
    plt.imshow(img2)
    plt.show()
    plt.imshow(imgClipped);
