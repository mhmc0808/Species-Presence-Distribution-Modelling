#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 18:50:10 2025
# /Users/zacharygavin/Library/spyder-6/envs/spyder-runtime/bin/python -m pip install <package>
@author: zacharygavin
"""
#  --- Library imports ---

import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, r2_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import random
import reverse_geocoder as rg
from netCDF4 import Dataset
from netCDF4 import num2date

import geodatasets




##### First we load in a geo data frame containing the geometries of the us states
wget.download("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip")
gdf = gpd.read_file('/Users/zacharygavin/Documents/AML-Group-work-/species/cb_2018_us_state_500k.zip')


# removes territories+non contiguous states 
# we'll use this later to reject pseudo-absences that are generated in the sea
gdf = gdf[~gdf['STUSPS'].isin({'PR', 'AK', 'AS', 'VI', 'HI', 'MP','GU'})]





#  --- Load the species observation data, df needs to have features already ---
obs_data = df.loc[df['species_id']==41301].reset_index(drop=True)
rest_data = df.loc[df['species_id']!=41301].reset_index(drop=True)


# Clean NAs
obs_data = obs_data.dropna(subset=["latitude", "longitude"])

max_lat = np.ceil(obs_data["latitude"].max())
min_lat = np.floor(obs_data["latitude"].min())
max_lon = np.ceil(obs_data["longitude"].max())
min_lon = np.floor(obs_data["longitude"].min())

# Geographic extent (minx, miny, maxx, maxy)
geographic_extent = (min_lon, min_lat, max_lon, max_lat)

# Slightly larger extent
buffer_factor = 1.25
lon_buffer = (max_lon - min_lon) * (buffer_factor - 1) / 2
lat_buffer = (max_lat - min_lat) * (buffer_factor - 1) / 2
sample_extent = (
    min_lon - lon_buffer,
    min_lat - lat_buffer,
    max_lon + lon_buffer,
    max_lat + lat_buffer,
)


# See which features are highly correlated
feature_corr = df[['density', 'temp_mean', 'temp_var',
                   'prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
                   'avg_ndvi', 'distance_to_coast', 'diversity', 'richness', 'evenness',
                   'State', 'County', 'Farmland Value']].corr(numeric_only=True)

pairs = np.array(np.where(abs(feature_corr) >.5) ).T
pairs = pairs[pairs[:,0] != pairs[:,1], :]
fcc0 = feature_corr.columns[pairs[:,0]]
fcc1 = feature_corr.columns[pairs[:,1]]

# This gives the cfeatures with correlations over 0.5
high_corr = np.column_stack((fcc0, fcc1))
high_corr




###############################################
########### --- ABSENCE GENERATION --- ########
###############################################

## We are doing this by generating a grid across the whole extent, and only accepting points that 
# fall within the geometry of the cont. US.

####### --- By uniform pseudo-absences --- #####
np.random.seed(1)
num_background = 1000 
acc_rate=0.5     # approx half of samples are over land

# Generate random points within extent
random_lons = np.random.uniform(sample_extent[0], sample_extent[2], int(num_background/acc_rate))
random_lats = np.random.uniform(sample_extent[1], sample_extent[3], int(num_background/acc_rate))

# Remove samples in extent but outside US (ie. Canada, Mexico or the ocean)
valid_samples = np.zeros(int(num_background/acc_rate), dtype=bool)
for s in range(0,int(num_background/acc_rate)): 
    valid_samples[s] = gdf.contains(Point(random_lons[s],random_lats[s])).any().astype(bool)
    
background = pd.DataFrame({"longitude": random_lons[valid_samples], "latitude": random_lats[valid_samples]})

### adding the temp data + precip data to background
background = load_full_dataset(background, '/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/AML Data')






######### --- By One-Vs-Rest --- ##########
rows = np.arange(0,rest_data.shape[0])
chosen_rows = np.random.choice(rows, 1000, replace=False) # could probably do replace=True here too
background = rest_data.iloc[chosen_rows, :]








# Presence data processed already using Michael's pipeline


#  --- Combine presence and absence data ---
presence = obs_data[['longitude', 'latitude','temp_mean', 'temp_var','prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
                     'avg_ndvi', 'distance_to_coast']].copy().dropna()
presence["pa"] = 1

absence = background[['longitude','latitude','temp_mean', 'temp_var','prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
                      'avg_ndvi', 'distance_to_coast']].copy()
absence["pa"] = 0


all_points = pd.concat([presence, absence], ignore_index=True)



################################################################################
############## ------- Do the actual classification ------- ####################
################################################################################

# features 
X = all_points.drop(['pa','latitude','longitude'], axis=1)

# labels
y= all_points['pa']


# train/validate split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=5)
 

# Do the desired LR/RF/GB method

model = LogisticRegression(max_iter=1000)

model = RandomForestClassifier(criterion='entropy', n_estimators=100, max_depth=20)
model = GradientBoostingClassifier(n_estimators=20, learning_rate=1,max_depth=1, random_state=0)


model.fit(X_train,y_train)


############################################
######## ---- Generating maps ---- #########
############################################

####### JOINING DATA TO A BACKGROUND GRID
# The following section ONLY NEEDS TO BE RUN ONCE PER ENVIRONMENT (it is not fast)
#-------------------------------------------------------------------
# Generating a grid for the whole of the US
x, y = np.meshgrid(np.linspace(sample_extent[0], sample_extent[2], 1000 ),np.linspace(sample_extent[1], sample_extent[3],1000), indexing='ij')
grid_init = pd.DataFrame({'longitude': x.ravel(), 'latitude':y.ravel()} )
# Only keep a pt it if its in continental US
valid_samples = np.zeros(grid_init.shape[0], dtype=bool)
for s in range(0, grid_init.shape[0]): 
    valid_samples[s] = gdf.contains(Point(grid_init['longitude'][s],grid_init['latitude'][s])).any().astype(bool)
    
    print(s)
    
back_grid = grid_init[valid_samples]

# fitting data to this grid using the functions
back_grid = load_full_dataset(back_grid, '/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/AML Data')
# ------------------------------------------------------------------


##### using this grid to make a BASIC MAP --------------------------
# Specialising down to the features we're currently using
back_predict = back_grid[['temp_mean', 'temp_var','prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
                      'avg_ndvi', 'distance_to_coast']]

# Need to fill in avg_nvdi NaN values for Logistic regression 
back_predict = back_predict.fillna(back_predict.mean(numeric_only=True))

# gives us flexibility to use a threshold
map_threshold = 0.1
preds = (model.predict_proba(back_predict)[:,1] > map_threshold).astype(int)


# putting coords in a df with predictions at that point
preds_df = pd.concat([back_grid[['longitude','latitude']], pd.Series(preds, name='preds')], axis=1)


# plotting the results on a map
preds_df['pred_label'] = preds_df['preds'].map({0: 'absence', 1: 'presence'})
ax=sns.scatterplot(
    data=preds_df,
    x='longitude',
    y='latitude',
    hue='pred_label',     # color by category
    palette={'absence': "tomato", 'presence': "royalblue"},     # categorical color palette
    s=1          # size of points
)

ax.legend(markerscale=5)

# overlaying the validation set presence observations
#vi=X_val.index
#vp = vi[vi<len(obs_data)]
#va = vi[vi>=len(obs_data)]-len(obs_data)

#plt.scatter(obs_data['longitude'][vp], obs_data['latitude'][vp], marker='+' , s=10, c='orange')
#plt.scatter(background['longitude'][va], background['latitude'][va], marker='+' , s=10, c='olivedrab')
plt.title("Plot of habitat prediction for Tadarida Brasiliensis")
plt.legend(title='Category')
plt.show()


######## Now produce a HEATMAP --------------------------------------

probs =  model.predict_proba(back_predict)[:, 1] 
probs_df = pd.concat([back_grid[['longitude','latitude']], pd.Series(probs, name='probs')], axis=1)


prob_grid = probs_df.pivot_table(
    index='latitude',
    columns='longitude',
    values='probs'
)


sns.heatmap(
    prob_grid.sort_index(ascending=False),
    cmap='coolwarm',
    cbar_kws={'label': 'Predicted Probability'},
    vmin=0,
    vmax=1,
    linewidths=0,       #remove lines between cells
    linecolor=None,     
)
plt.title('Probability heatmap for Tadarida Brasiliensis')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xticks([], [])
plt.yticks([], [])
plt.show()





##########################################################
###### --- Fit a model for each species at once --- #####
##########################################################

# this is now all in modelFitting.py

ids = df.species_id.value_counts(sort=False)

# arbitrarily chose 1000 as the number of observations needed for a model
mod_species = ids[ids>=200]
species_ids = mod_species.index.tolist()

#### - Create and fit a model for each species, storing these in a list ------
species_models = []
for species in species_ids:
    
    # load data in for that species
    obs_data = df.loc[df['species_id']==species].reset_index(drop=True)
    obs_data = obs_data.dropna(subset=["latitude", "longitude"])
    
    presence = obs_data[['longitude', 'latitude','temp_mean', 'temp_var','prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
                         'avg_ndvi', 'distance_to_coast']].copy().dropna()
    presence["pa"] = 1
     
    # absences stay the same for each species
    all_points = pd.concat([presence, absence], ignore_index=True)
    
    # labels, features, and train/validate split
    X = all_points.drop(['pa','latitude','longitude'], axis=1)
    y= all_points['pa']
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=5)
    
    
   # model = GradientBoostingClassifier(n_estimators=20, learning_rate=1,max_depth=1, random_state=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    species_models.append([species, model, X_val, y_val])


all_models = [x[1] for x in species_models]
all_val =  [x[2:4] for x in species_models]



############ - make predictions for each species  -------------------------
n = len(species_ids)
preds = [None] * n
probs = [None] * n
for s in range(0,n):
    preds[s] = all_models[s].predict(all_val[s][0])
    probs[s] = all_models[s].predict_proba(all_val[s][0])



########### - Analysing the coefficients, this DID NOT COME TO ANYTHING ---------------------------
features = X.columns

coefs = [None] * n
for s in range(0, n):
    coefs[s] = all_models[s].coef_

coefs = np.array(coefs)

coefs_df = pd.DataFrame(coefs.squeeze(axis=1), columns=features)

# for instance, try seeing if theres similarity between species that like the coast    
plt.scatter(coefs_df['distance_to_coast'], coefs_df['temp_mean'], )
plt.xlabel("distance to coast")
plt.ylabel("mean temperature")
plt.show()

coast_creatures = coefs_df[coefs_df['distance_to_coast']< -0.00001].index

mask = np.isin(df['species_id'], [species_ids[i] for i in coast_creatures])  # wrong since need to translate to actual species ids
cc_names = df.loc[mask]['species_name']
np.unique(cc_names)
# (there isn't, really) 


### PCA/Clustering of species FOR LOGISTIC Regression

coefs_scaled = StandardScaler().fit_transform(coefs_df)
pca = PCA(n_components=2)

coef_pca = pca.fit_transform(coefs_scaled)

print(pca.explained_variance_ratio_)
print( sum(pca.explained_variance_ratio_))

plt.scatter(coef_pca[:,0], coef_pca[:,1], )

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(coefs_scaled)

labels = kmeans.labels_

plt.scatter(coef_pca[:, 0], coef_pca[:, 1], c=labels, cmap='tab10', s=30)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
plt.title("K-Means Clustering (k=4)")
plt.show()

# unfortunately, I don't think there's much signal here 



###################################################
######### ------- Model Selection -------- ########
###################################################

# Now we'll do some model selection - this is all done on the validation set 

# Want to make predictions for each of the points in the test data using the model, 
# and then compute the false postive and false negative rates 

threshold = 0.5
val_preds = (all_models[61].predict_proba(all_val[61][0])[:,0]<threshold).astype(int)
cm = confusion_matrix(y_val, val_preds)

print('True positive:', cm[1,1],
      '\nFalse negative:', cm[1,0],
      '\nFalse positive:', cm[0,1],
      '\nTrue negative', cm[0,0])




### APPROACH 1: Sensitivity + Specificity


# prob of postive prediction, given truly positive
sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])

# prob of negative prediction, given truly negative
specificity = cm[0,0]/(cm[0,0]+cm[0,1])


print('sensitivity:', sensitivity, 
      '\nspecificity:', specificity)


# try a bunch of possible thresholds to find the one that maximises sensitivity+specificity
n = 100
thresh_values = np.linspace(0,1, n)
sens_spec = np.zeros(n)
for i in range(n):
    
    val_preds = (all_models[61].predict_proba(X_val)[:,0]<thresh_values[i]).astype(int)
    cm = confusion_matrix(y_val, val_preds)
    sensitivity = cm[1,1]/(cm[1,1]+cm[1,0])
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    
    sens_spec[i] = sensitivity + specificity
    
    
# look graphically for the maximum 
plt.scatter(thresh_values, sens_spec)

# find the numeric value of the maximum
opt_thresh = thresh_values[np.where(sens_spec == max(sens_spec))]
opt_thresh    





# APPROACH 2: ROC + AUC ROC

fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:,1])

plt.plot(fpr, tpr)

auc_roc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
auc_roc









#### the following is all done in modelFitting

#######################################################################
##### ----- Testing on presence/absence test data provided ----- ######
#######################################################################

# To test, we can use the presence/absence data provided in the test set

# load data in
data_test = np.load('/Users/zacharygavin/Documents/AML-Group-work-/species/species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']    # 2D array, rows are number of datapoints 
                                      # and columns are "latitude" and "longitude" 
# data_test['test_pos_inds'] is a list of lists, where each list corresponds to 
# the indices in test_locs where a given species is present, it can be assumed 
# that they are not present in the other locations 
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))    

#
species =  41301
presence = np.zeros(len(test_locs))
presence[test_pos_inds[species]] = 1
test_df = pd.DataFrame({"latitude": test_locs[:,0], "longitude": test_locs[:,1], 
                        'presence':presence})

# Remove samples in extent but outside US (ie. Canada, Mexico or the ocean)
valid_samples = np.zeros(len(test_locs), dtype=bool)
for s in range(0,len(test_locs)): 
    valid_samples[s] = gdf.contains(Point(test_locs[s,1], test_locs[s,0])).any().astype(bool)


# New DataFrame us_test has cols [long,lat,pres/abs], for each point in the US
us_test = test_df.loc[valid_samples]

# Add features and creating feature matrix and targets
us_test_wf = load_full_dataset(us_test, '/Users/zacharygavin/Documents/AML-Group-work-/AML_Project/AML Data')                                                                        
X_test = us_test_wf[['temp_mean', 'temp_var','prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
                      'avg_ndvi', 'distance_to_coast']]

# make a list of the vectors of presences for each species
test_ys = [None] * n
for s in range(0,n):
    
    test_ys[s] = presence
    

# Then we can calculate an AUC-ROC for every species at once, and average
    
    
    


######### Visualising the Test Data -------------------------


plt.plot(us_test.loc[us_test.presence==1].longitude, us_test.loc[us_test.presence==1].latitude,
         'b.', label='presence', ms=1)
plt.plot(us_test.loc[us_test.presence==0].longitude, us_test.loc[us_test.presence==0].latitude, 
         'r.', label='absence',ms=1)

plt.xlim(sample_extent[0]+2, sample_extent[2]+5)  
plt.ylim(sample_extent[1]+1, sample_extent[3]+3)  
plt.title("True range of Tadarida Brasiliensis")
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.legend(markerscale=5)



######## ------ Evalutating the models on test data ------- ##########

n = len(species_ids)
test_preds = [None] * n
test_probs = [None] * n
auc_rocs = [None] * n 
for i in range(0,n):
    
    test_preds[i] = all_models[i].predict(X_test)
    test_probs[i]  = all_models[i].predict_proba(X_test)
    auc_rocs[i] = auc_roc = roc_auc_score(y_test, all_models[i].predict_proba(X_test)[:,1])
    

    
    













    
    
    
    
    
    



