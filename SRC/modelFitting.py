#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:22:37 2025

"""


import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
#import rasterio
#from rasterio.mask import mask
#from rasterio.plot import show
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, make_scorer, log_loss
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, r2_score, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
import random
import reverse_geocoder as rg
from netCDF4 import Dataset
from netCDF4 import num2date
import wget
import geodatasets
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from sklearn.model_selection import GridSearchCV


from sklearn.model_selection import cross_val_score


datadir = "/Users/michael/Desktop/AML Repo/AML-Group-work-/AML Data"

#datadir = "C:/Users/micha/Desktop/AML Group Projec/AML-Group-work-/AML Data"


# Loading training Data 

d = np.load(datadir + "/" + "species_train.npz", allow_pickle=True)

df = pd.DataFrame(d["train_locs"], columns=["latitude", "longitude"])
df["species_id"] = d["train_ids"]
species_names = dict(zip(d["taxon_ids"], d["taxon_names"]))
df["species_name"] = df["species_id"].map(species_names)




df = load_full_dataset(df, datadir)

gdf = gpd.read_file(datadir + "/" + 'cb_2018_us_state_500k.zip')

# removes territories+non contiguous states 
# we'll use this later to reject pseudo-absences that are generated in the sea
gdf = gdf[~gdf['STUSPS'].isin({'PR', 'AK', 'AS', 'VI', 'HI', 'MP','GU'})]





def generatePsuedoAbsence(df):
    obs_data = df.dropna(subset=["latitude", "longitude"])
    
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
    
    
    
     # Generate pseudo-absence points
    
    # We are doing this by generating a grid across the whole extent, and only accepting points that 
    #fall within the geometry of the cont. US.
    
    ######### By uniform pseudo-absences
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
    background = load_full_dataset(background, datadir)
    return background 


def generatePerSpeciesPsuedoAbsence(df,speciesID, feature_cols, background): 
    obs_data = df.loc[df['species_id']==speciesID].reset_index(drop=True)
    rest_data = df.loc[df['species_id']!=speciesID].reset_index(drop=True)
          
    
    #  Combine presence and absence data 
    
    cols = ['latitude', 'longitude'] + feature_cols
          
    
    presence = obs_data[cols].copy().dropna()
    presence["pa"] = 1
    
    absence = background[cols].copy()
    absence["pa"] = 0
    
    all_points = pd.concat([presence, absence], ignore_index=True)
    
    
    X = all_points.drop(['pa','latitude','longitude'], axis=1)
    
    # labels
    y= all_points['pa']
    return X, y 

def generatePerSpeciesOneVRest(df, speciesID, feature_cols):
    # Load  species observation data 
    obs_data = df.loc[df['species_id']==speciesID].reset_index(drop=True)
    rest_data = df.loc[df['species_id']!=speciesID].reset_index(drop=True)
      
    ######### By One-Vs-Rest
    rows = np.arange(0,rest_data.shape[0])
    chosen_rows = np.random.choice(rows, 1000, replace=False) # could probably do replace=True here too
    background = rest_data.iloc[chosen_rows, :]

    
    
    
    #   Combine presence and absence data 
    
    cols = ['latitude', 'longitude'] + feature_cols
          
    
    presence = obs_data[cols].copy().dropna()
    presence["pa"] = 1
    
    absence = background[cols].copy()
    absence["pa"] = 0
    
    all_points = pd.concat([presence, absence], ignore_index=True)
    
    
    X = all_points.drop(['pa','latitude','longitude'], axis=1)
    
    # labels
    y= all_points['pa']
    return X, y 


feature_cols = [
       'temp_mean', 'temp_var',
       'prcp_mean', 'prcp_var', 'avg_wind', 'soil_mean', 'soil_var',
       'avg_ndvi', 'distance_to_coast', 'Farmland Value']



### generate background data for psuedoabsence for training set 

background = generatePsuedoAbsence(df)



############# load test data #########

data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']    # 2D array, rows are number of datapoints 
                                      # and columns are "latitude" and "longitude" 
                                      
test_df = pd.DataFrame({
    "latitude": test_locs[:, 0],
    "longitude": test_locs[:, 1],
})

# keep original row indices
test_df["orig_index"] = np.arange(len(test_df))

# Combine the testset with the additional feature cols 
test_df = load_full_dataset(test_df, datadir)

test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))  

valid_samples = test_df.apply(
    lambda row: gdf.contains(Point(row.longitude, row.latitude)).any(), axis=1
)
us_test = test_df.loc[valid_samples]
                                         

ids = df.species_id.value_counts(sort=False)


# filter only on species with at least 1000 obeservation in train set 

mod_species = ids[ids>=1000]

species_ids = mod_species.index.tolist()





### Create and fit a model for each species and store results as DF #####

accuracydf = pd.DataFrame()

feature_importance_df = pd.DataFrame()
for species in species_ids:
    
    species_name = str(df.loc[df['species_id'] == species]['species_name'].unique()[0])

    ############# load train data #########
    
    
    ## Uncomment depending on if one vs rest or psuedoabsence 
    
    #X, y =  generatePerSpeciesOneVRest(df, species , feature_cols)
    
    X, y = generatePerSpeciesPsuedoAbsence(df,species, feature_cols, background)
    
    
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, test_size=0.2, random_state=5)
     
    
    ############# Generate models #########
    
    # Uncomment depending on which model needs to be run 
    
    ## Logistic Regression 
    
    #model = LogisticRegression()
    
    
    # logistic Regression with Gridsearrch 
    #lRmodel = LogisticRegression(max_iter = 500)
    
    # param_grid_LR = {
    # 'penalty': ['l1', 'l2'],
    # 'C': [0.01, 0.1, 1, 10],
    # 'solver': ['liblinear', 'saga'],
    # }
    # model = GridSearchCV(lRmodel, param_grid_LR, scoring = 'roc_auc', cv=5)


    ### Random Forest ####
    #model = RandomForestClassifier(criterion='entropy', max_depth=3, random_state=1000)
    
    
    # RF with Gridsearch
    # rfmodel = RandomForestClassifier()
    # param_grid_RF = {
    # 'n_estimators': [50, 100, 200],
    # 'max_depth': [3, 5, 10],
    # 'max_features': ['auto', 'sqrt'],
    # }
    
    # model = GridSearchCV(rfmodel, param_grid_RF, scoring = 'roc_auc', cv=5)
    
    ######## Gradient boosting ######
    
    # model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    #     max_depth=1, random_state=0)
    
    
    gbModel = GradientBoostingClassifier()
    
    param_grid_GBC = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [1, 3, 5, 7],
    }
    
    
    
    model = GridSearchCV(gbModel, param_grid_GBC, scoring = 'roc_auc', cv=5)
    
    
    # fit models 
    
    model.fit(X_train, y_train)
      
    params = model.best_params_
    #params =  'base'
    
    
    # presence indices from  the test data set
    
    pos_inds = test_pos_inds[species]
    
    # compute presence for filtered dataset
    us_test["presence"] = us_test["orig_index"].isin(pos_inds).astype(int)
    
    
                                   
    map_threshold = 0.5
    
    X_test = us_test[feature_cols]
    y_test = us_test[['presence']]
    
    
    # sometimes test set has no species absence obeservation so need to skitop this 
    if len(np.unique(y_test)) < 2:
        print(f"Skipping {species_name}: test set has only a single class")
        continue
    test_pred = (model.predict_proba(X_test)[:,1] > map_threshold).astype(int)
    
    
    # Compute evaluation metrics and append to the df 
    
    auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    accuracy = accuracy_score(y_test,test_pred)
    
    logloss = log_loss(y_test, model.predict_proba(X_test)[:,1])
   
    
    
    
    
    # importance_dict = {
    # column: model.feature_importances_[i]
    # for i, column in enumerate(X_train.columns)
    # }
    
    # # Convert to a one-row DataFrame
    # importance_df = pd.DataFrame([importance_dict])
    # importance_df['species'] = species
    # importance_df['species_name'] = species_name
    # feature_importance_df = pd.concat([feature_importance_df, importance_df])


    
    tempDf = pd.DataFrame({
    "Species": [species_name],
    "Species ID": [species],
    "auc_roc": [auc_roc],
    "accuracy": [accuracy],
    "Model":  [str( model)],
    "Params": [str(params)],
    "logloss": [logloss]
    
    
    })
    accuracydf = pd.concat([accuracydf, tempDf]) 
    
    
    

# save the results as a csv, we manually change the csv name for each model and training type 

accuracydf.to_csv('GB tuned psuedo abesence accuracy.csv')    
    


################################# Plotting results for all models ########################

# Plot feature importance if Random Forrest is being used 
    
feature_importance_df_melted = pd.melt(feature_importance_df, id_vars = ['species', 'species_name'])


plt.figure(figsize=(12, 6))

sns.boxplot(
    data=feature_importance_df_melted,
    x="variable",
    y="value",
    palette=custom_palette,
    width=0.6,
    linewidth=1.2
)

plt.xticks(rotation=45, ha='right')

plt.xlabel("Features", fontsize=12)
plt.ylabel("Normalised Total Reduction of Entropy", fontsize=12)

plt.title("Distribution of Feature Importances", fontsize=14)

plt.tight_layout()

plt.show()



####### THe below is just to produce the graph in our report where we, load and process each of the csv's so we can graph. 
    


GBtunedaccruracy1 = pd.read_csv('GB tuned onevrest accuracy.csv')
GBbaseaccruracy1 = pd.read_csv('GB base onevrest accuracy.csv')

GBbaseaccruracy1 = pd.read_csv('base cross entropy AUC gb.csv')
GBbaseaccruracy1= GBbaseaccruracy1.rename(columns = {'cross-entropy': 'logloss'})

GBtunedaccruracy1= pd.read_csv('GB cross entropy one v rest.csv')

GBtunedaccruracy1= GBtunedaccruracy1.rename(columns = {'cross-entropy': 'logloss'})



GBtunedaccruracyPsuedo = pd.read_csv('GB tuned psuedo abesence accuracy.csv')

GBtunedaccruracyPsuedo = GBtunedaccruracyPsuedo[GBtunedaccruracyPsuedo['Params'] != 'base']


GBbaseaccruracyPsuedo = pd.read_csv('GB base psuedo abesence accuracy.csv')






RFtunedaccruracy1 = pd.read_csv('RF Tuned onevrest accuracy.csv')
RFbaseaccruracy1 = pd.read_csv('RF Base onevrest accuracy.csv')
RFtunedaccruracyPsuedo = pd.read_csv('RF Tuned psuedo abesence accuracy.csv')
RFbaseaccruracyPsuedo = pd.read_csv('RF Base psuedo abesence accuracy.csv')

LRtunedaccruracy1 = pd.read_csv('LR Tuned onevrest accuracy.csv')
LRbaseaccruracy1 = pd.read_csv('LR Base onevrest accuracy.csv')
LRtunedaccruracyPsuedo = pd.read_csv('LR Tuned psuedo abesence accuracy.csv')
LRbaseaccruracyPsuedo = pd.read_csv('LR Base psuedo abesence accuracy.csv')


GBpsuedo = pd.concat([GBtunedaccruracyPsuedo, GBbaseaccruracyPsuedo])
GBpsuedo = GBpsuedo.drop(columns = ['Unnamed: 0', 'accuracy'])
GBpsuedo['Training'] = 'Pseudo Absence'

GBonevrest = pd.concat([GBtunedaccruracy1, GBbaseaccruracy1])
GBonevrest = GBonevrest.drop(columns = ['Unnamed: 0', 'accuracy'])
GBonevrest['Training'] = 'One vs Rest'

GBdf =pd.concat([GBpsuedo, GBonevrest])
GBdf['hyperparams'] = np.where(GBdf['Model'] == 'GradientBoostingClassifier(learning_rate=1.0, max_depth=1, random_state=0)', 'Base', 'Tuned')
GBdf= GBdf.dropna()



RFpsuedo = pd.concat([RFtunedaccruracyPsuedo, RFbaseaccruracyPsuedo])
RFpsuedo = RFpsuedo.drop(columns = ['Unnamed: 0', 'accuracy'])
RFpsuedo['Training'] = 'Pseudo Absence'

RFonevrest = pd.concat([RFtunedaccruracy1, RFbaseaccruracy1])
RFonevrest = RFonevrest.drop(columns = ['Unnamed: 0', 'accuracy'])
RFonevrest['Training'] = 'One vs Rest'

RFdf =pd.concat([RFpsuedo, RFonevrest])
RFdf['hyperparams'] = np.where(RFdf['Model'] == "RandomForestClassifier(criterion='entropy', max_depth=3, random_state=1000)", 'Base', 'Tuned')
RFdf= RFdf.dropna()



LRpsuedo = pd.concat([LRtunedaccruracyPsuedo, LRbaseaccruracyPsuedo])
LRpsuedo = LRpsuedo.drop(columns = ['Unnamed: 0', 'accuracy'])
LRpsuedo['Training'] = 'Pseudo Absence'

LRonevrest = pd.concat([LRtunedaccruracy1, LRbaseaccruracy1])
LRonevrest = LRonevrest.drop(columns = ['Unnamed: 0', 'accuracy'])
LRonevrest['Training'] = 'One vs Rest'

LRdf =pd.concat([LRpsuedo, LRonevrest])
LRdf['hyperparams'] = np.where(LRdf['Model'] == 'LogisticRegression()', 'Base', 'Tuned')
LRdf= LRdf.dropna()




#### plot the log loss 

f, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True) 
hue_order = ["Tuned", "Base"]
custom_palette = ["purple", "#66c2a5"]
sns.set_theme(style="whitegrid")

sns.boxplot(
    data=GBdf,
    x="hyperparams",
    y="logloss",
    hue='Training',
    palette=custom_palette,
    width=0.6,
    linewidth=1.2,
    ax=axes[0]
)
axes[0].set_title("Gradient Boosting", fontsize=12)
axes[0].set_xlabel("")
axes[0].set_ylabel("Cross Entropy")

sns.boxplot(
    data=RFdf,
    x="hyperparams",
    y="logloss",
    hue='Training',
    palette=custom_palette,
    width=0.6,
    linewidth=1.2,
    ax=axes[1]
)
axes[1].set_title("Random Forest", fontsize=12)
axes[1].set_xlabel("Tuned Vs Base Model")

sns.boxplot(
    data=LRdf,
    x="hyperparams",
    y="logloss",
    hue='Training',
    palette=custom_palette,
    width=0.6,
    linewidth=1.2,
    ax=axes[2]
)
axes[2].set_title("Logistic Regression", fontsize=12)
axes[2].set_xlabel("")

for ax in axes:
    ax.get_legend().remove()

handles, labels = axes[0].get_legend_handles_labels()
f.legend(
    handles,
    labels,
    loc='center left',
    bbox_to_anchor=(1.02, 0.5), 
    title="Training Type",
    fontsize=10,
    title_fontsize=11,
    frameon=True
)

plt.tight_layout(rect=[0, 0, 0.995, 1])  
plt.show()




# looking at AUC vs Log loss

custom_palette = ["purple", "#66c2a5"]
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=LRbaseaccruracy1['auc_roc'], 
    y=LRbaseaccruracy1['logloss'],
    s=90,
    alpha=0.8,
    edgecolor="black",
    color=custom_palette[0]  # select one color from your palette
)

plt.title("AUC-ROC vs Log Loss for Base Logisitic Regression", fontsize=14, weight="bold")
plt.xlabel("AUC-ROC", fontsize=12)
plt.ylabel("Log Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

