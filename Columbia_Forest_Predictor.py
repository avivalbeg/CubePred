from RasterClass import *
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cluster import KMeans
from scipy.signal import convolve2d,convolve
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

#This is for elegance. The warnings should eventually be heeded.
import warnings
warnings.filterwarnings("ignore")

class Columbia_Forest_Predictor:
    
    def __init__(self,mode='auto',filename = None):
        dc = datacube.Datacube(app    = "my_random_app",
                       config = "/home/localuser/.datacube.conf")
        #changing these parameters leads to over/under fitting.
        self.clf = RF(max_depth=10,n_estimators=100, random_state=0,warm_start=True)
        
        if mode == 'auto': 
            dataset,img,labels = self.plot_and_retrieve()
        
        elif mode == 'load':
            try:
                [self.clf,self.keychain] = pickle.load(open(filename, 'rb'))
            except:
                raise('No such file exists.')
                
            
        elif mode == 'shape_folder':
            done = False
            self.DataFrame = gpd.GeoDataFrame()
            
            while not done: 
                try:
                    folder = input('what is the folder name?')
                    files = self.read_folder(folder)
                    
                    for elem in files:
                        print(elem)
                        
                    done = input("Is this correct?(\'Y\' = Yes, \'N\' = No)")
                    
                    if done == 'Y':
                        done = True
                    elif done == 'N':
                        done = False
                        
                except:
                    print('not a valid folder.')
            #Yes this is clunky
            if folder == '.':
                folder = ''
                
            self.read_shapes(files)
            
            print(self.DataFrame.keys())
            
            Interest = input('What is our variable of interest?')
            print('The set of all such variable values is:\n')
            print(set(self.DataFrame[Interest]),'\n')
            
            done = False
            
            print('which of these do you want to include in the classification?\n')
            
            self.good_vals = []
            
            while not done:
                a = input()
                
                if a == 'Done':
                    done = True 
                    break
                    
                print('You want to include %s?(\'Y\' = Yes, \'N\' = No,\'Done\' = Done)'%str(a))
                
                check = input()
                
                if check == 'Y':
                    self.good_vals.append(int(a))
            while True:
                print('Do you want  to see the pictures as they are trained?(Y = Yes, N = No)')        
                check = input()
                if check == 'Y':
                    pictures = True
                    break
                elif check == 'N':
                    pictures = False
                    break
            
            print('making list of all shapes...')
            
            self.goodShapes = {}
            self.keychain = {}
            
            for i,good_val in enumerate(self.good_vals):
                self.keychain[good_val] = i+1
                self.goodShapes[good_val] =self.DataFrame[self.DataFrame[Interest] == good_val].geometry
                
            print('Done!')
            print('List of all relevant areas:')
            print(self.goodShapes)
            print('Training...')
            imgClippedTot = np.array([])
            LabelMatTot = np.array([])
            for Label in self.goodShapes:
                shapes = self.goodShapes[Label]
                for shape in shapes:
                    if shape == None:
                        continue
                    [_,LabelMat,_,imgClipped] = clip_from_shape(shape)
                    #if you want to display the pictures
                    if pictures:
                        plt.imshow(imgClipped)
                        plt.show()
                        plt.imshow(LabelMat)
                        plt.show()
                    else: #this is to be a progress bar
                        pass
                    LabelMat = LabelMat*self.keychain[Label]
                    #Below is not efficient. 
                    imgClippedTot = np.append(imgClippedTot, imgClipped.flatten())
                    LabelMatTot = np.append(LabelMatTot,LabelMat.flatten())
            #print(LabelMatTot)
            #print(imgClippedTot)
            self.train_forest(imgClippedTot,LabelMatTot)
            print('Done!')
            
            
            
                
        #self.clf = RF(max_depth=4, random_state=0)
        #self.clf = train_forest(img,labels)
        
    def read_folder(self,folder):
        all_files = os.listdir(folder)
        shp_files = []
        for file in all_files:
            if file.endswith(".shp"):
                shp_files.append(os.path.join(os.getcwd()+'/'+folder+'/', file))
        return shp_files
    
#     def train_forest_shapes(self,data_label,data_value):
#         polys = self.shapes[self.shapes[data_label]==data_value]
#         for elem in polys:
            
    
    def read_shapes(self,files):
        print('reading files.')
        for file in files:
            self.DataFrame =self.DataFrame.append(gpd.read_file(file))
        print('done reading files')
            
    def is_l7_dataset_clean(dataset):    
        clear_pixels = dataset.pixel_qa.values == 2 + 64
        water_pixels = dataset.pixel_qa.values == 4 + 64
        a_clean_mask = np.logical_or(clear_pixels, water_pixels)
        return a_clean_mask
    
    def composite(ds):
        ds = ds.where(is_l7_dataset_clean(ds))
        return ds.median(dim='time', skipna=True, keep_attrs=False)

    def pull_Median(self,dataset):
        median_dataset = composite(dataset)
        img = np.dstack((median_dataset.swir1.values/np.nanmax(median_dataset.swir1.values),median_dataset.nir.values/np.nanmax(median_dataset.nir.values),median_dataset.red.values/np.nanmax(median_dataset.red.values)))
        img = np.nan_to_num(img,-999)
        return img

    def make_Labels(self,img,n_clusters=5,style = 'KMeans'):
        a2 = np.ones((3,3))
        img = np.nan_to_num(img,-1)
        img_swir = convolve(a2,img[:,:,0])
        img_nir = convolve(a2,img[:,:,1])
        img_grn = convolve(a2,img[:,:,2])
        img2 = np.dstack((img_swir/np.max(img_swir),img_nir/np.max(img_nir),img_grn/np.max(img_grn)))
        img2_flat = img2.reshape(-1,3)   
        if style == 'KMeans':
            classifier3 = KMeans(n_clusters=3).fit(img2_flat)
        else:
            print('different')
            classifier3 = self.clf
        labels = classifier3.predict(img2_flat).reshape(img2.shape[:2])
        labels = np.nan_to_num(labels,-999)
        return img2,labels
    # import sys
    def get_img(self,dataset,index):
        
        img = np.dstack((dataset.where(dataset > 0).isel(time = index).swir1.values/np.nanmax(dataset.where(dataset > 0).isel(time = index).swir1.values),dataset.where(dataset > 0).isel(time = index).nir.values/np.nanmax(dataset.where(dataset > 0).isel(time = index).nir.values),dataset.where(dataset > 0).isel(time = index).red.values/np.nanmax(dataset.where(dataset > 0).isel(time = index).red.values)))
        return img
    
    def plot_and_retrieve(self,x=(1.65,1.73),y=(-73.7,-73.4),crs = 'EPSG:4326', plot = True,time =('2012-01-01', '2015-01-01'), falseColor = ["FIX THIS LATER"],style= 'KMeans'):
        print('loading data!')
        dataset = dc.load( product = "ls7_ledaps_sanagustin",latitude = x,longitude = y,crs = crs,measurements = ['red', 'green', 'blue', 'nir', 'swir1', 'swir2', 'pixel_qa'],time = time)
        print('Done Loading data')
        if plot == True:

    #         if "matplotlib.pyplot" not in sys.modules:

    #             print("matplotlib not imported... Importing.")

    #             import matplotlib.pyplot as plt

    #             print('Done importing.')
            print('Beginning plotting') 
            for index in range(len(dataset.time)):  
                img = self.get_img(dataset,index)
                plt.figure(figsize=(12,4))
                plt.title(dataset.time[index].values)
                plt.imshow(img)
                plt.show()
            finim = self.pull_Median(dataset)
            plt.figure(figsize=(12,4))
            plt.title("Composite Image")
            plt.imshow(finim)
            plt.show()
            finim,labels = self.make_Labels(finim,style = style)
            plt.figure(figsize=(12,4))
            plt.title("Labels")
            plt.imshow(labels)
            plt.show()
            return dataset,finim,labels
        return dataset
    
    def make_Labels_From_Dataset(self,dataset):
        finim = self.pull_Median(dataset) # runs median composite on a dataset
        _,labels = self.make_Labels(finim,style = 'different')
        return finim,labels
        
        
    
    def train_forest(self,img,labels):
        #clf = RF(max_depth=4, random_state=0)   #Make the Random Forest
        img = np.array(img)
        labels = np.array(labels)
        labels_Flat = labels.flatten()
        img_Flat =img.reshape(-1,3)
#         k=0
#         for i in img_Flat.flatten():
#             if np.isnan(i):
#                 k+=1
#         print(k)
        #print(img_Flat)
        #print(labels_Flat)
        self.clf.fit(img_Flat,labels_Flat)

    def predict(self,img,plot =True):
        labels = self.clf.predict(img.reshape(-1,3))
        labels = labels.reshape(img.shape[0:2])
        if plot == True:
            plt.figure(figsize=(12,4))
            plt.imshow(img)
            plt.show()
            plt.figure(figsize=(12,4))
            print(labels.shape)
            plt.imshow(labels)
            plt.show()
        return labels

    def save_classifier(self,filename):
        pickle.dump([self.clf,self.keychain], open(filename, 'wb'))
        

    import xarray as xr
    def categorize(self,dataset):
        [_,d3] = self.clf.make_Labels_From_Dataset(dataset)
        cats = np.max(d3)
        d3A = []
        for i in range(int(cats)+1):
            d3i= xr.DataArray(d3==i,dims=['latitude','longitude'])
            if i!=0:
                for Id,val in self.keychain.items():
                    if val == i:
                        t = Id
                        break
            else:
                t='Unknown'
            d3i=d3i.to_dataset(dim=str(t))
            d3A.append(d3i)
        D3 = xr.merge(d3A)
        return D3
        
    
