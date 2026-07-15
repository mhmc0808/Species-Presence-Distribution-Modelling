Code running instructions:

All Python Libraries Required
- `numpy`
- `pandas`
- `zipfile`
- `matplotlib`
- `seaborn`
- `reverse_geocoder`
- `cartopy`
- `scipy`
- `sklearn`
- `geopandas`
- `seaborn`
- `rasterio`
- `shapely`
- `random`
- `netCDF4`
- `geodatasets`
- `os`
- `wget`
- `IPython`
- `graphviz`

Declaration of Generative AI Use:
In constructing our report, generative AI was used to aid us in loading and formatting external data into our code space. However, it was not used for any code generation or
report-writing.



FOLDERS:

Code Folder:

1. `dataloadingfunction.py`
   - Subsets our data to only continental US.
   - Loads all additional data gathered
   - Handles all data preprocessing and feature engineering, merging data to latitude and longitude in either train or test set.
   - Attaches environmental features to species data by latitude/longitude coordinates using KD-tree nearest-neighbour matching.
   - Calculates each species data distance to nearest coastline, proportion of farmland by county.

2. `human_pop_initial_eda.py`
   - training/test Data Info
   - model plausibility
   - human population analysis by comparing frequency of species data to US city population data.
     
3. `modelFitting.py`
   - loads full dataset from `dataloadingfunction.py`, also requires data filecb_2018_us_state_500k.zip.
   - generates pseudo-absence data by one-vs-rest or uniform random method.
   - for each species with more than 1000 observations, modelfitting:
     - builds training data
     - trains model using various ML models
     - performs GridSearch for hyperparameter tuning
     - evaluates model performance on test set (using AUC-ROC, accuracy, log-loss).
   - all results are saved to csv files stored in `Results` folder, where the name specifies the pseudo-absence data collection method and ML model used.
   - to change the model and the type of training data (one vs rest and Psuedabsence), we must comment and uncomment the appropriate lines of code. 

4. `SDM.py`
   - produces prediction maps across the US for different species, indicating where our models determine these species are likely to be present.
  
5. `Analysis.py`
   - conducts hypothesis tests
   - both for difference between AUC-ROC of models trained on uniform and one-vs-rest pseudo-absences
   - and for difference between  tuned and untuned models.
     

Data Folder: 

1. `species_train.npz`
   - contains training dataset (presence-only), where
     - train_locs - has [latitude, longitude] of each observation
     - train_ids - species ID for each observation
     - taxon_ids - list of unique species IDs present
     - taxon_names - species names corresponding to taxon_ids

2. `species_test.npz`
   - contains testing dataset (presence and absence data), where
     - test_locs - has [latitude, longitude] coordinates of where predictions are made
     - taxon_ids - list of unique species IDs included in test set
     - taxon_names - species names corresponding to taxon_ids
     - test_pos_inds - list of indices in test_locs for each species determining where the species is present. All other locations for that species are considered absent.
    
3. `farmland_proportion_by_county.csv`
   - contains details on farmland for all US counties.
   - in our analysis, we only extract State, County, Value (state and county to match with our data, value tells us the proportion of the given county that is farmland)
   - taken from USDA National Agricultural Statistics Service website: https://quickstats.nass.usda.gov/results/7A20CA20-5831-3B08-B570-788F9B3157BE
   - In order to extract this dataset, go to the USDA Quick Stats website: https://quickstats.nass.usda.gov/ and choose the following options: Group: FARMS & LANDS & ASSETS, Commodity: FARM OPERATIONS, Data Item: FARM OPERATIONS - AREA OPERATED, MEASURED IN PCT OF TOTAL LAND. Geographic Level: COUNTY, then click Get Data.
  
4. `nclimgrid_prcp.nc`
   - 2020 Monthly Averages precipitation (mm)
   - Source: National Centers for Environmental Information (NCEI)
   - Used to extract mean and variance precipitation grid data ranging across the continental US
     
5. `nclimgrid_tavg.nc`
   - 2020 Monthly Averages temperature (degrees celsius)
   - Source: National Centers for Environmental Information (NCEI)
   - Used to extract mean and variance temperature grid data ranging across the continental US
     
6. `simplemaps_uscities_basicv1.91.zip`
    -Data on population density of US cities, used for human bias analysis

7. `soilw.0-10cm.gauss.2020.nc`
   - 2020 4x Daily Volumetric Soil Moisture 10cm below ground
   - Source: National Centers for Environmental Information (NCEI)
   - Used to extract mean and variance soil moisture data ranging across the continental US
     
8. `uscities.csv`
    -Data on population density of US cities, used for human bias analysis. Extracted from `simplemaps_uscities_basicv1.91.zip` folder.

9.. `uwnd.10m.gauss.2020.nc`
   - 2020 10m above ground (m/s) average
   - Source: National Centers for Environmental Information (NCEI)
   - Used to extract mean wind speed and mean direction grid data ranging across the continental US

10. `VIIRS-Land_v001_NPP13C1_S-NPP_20200101_c20240126161431.nc`
   - 2020 Daily Averages, derived from the NOAA Climate Data Record (CDR)
   - Source: National Centers for Environmental Information (NCEI)
   - Used to extract NDVI grid data ranging across the continental US
     
11. `vwnd.10m.gauss.2020.nc`
   - 2020 10m above ground (m/s) average
   - Source: National Centers for Environmental Information (NCEI)
   - Used to extract mean wind speed and mean direction grid data ranging across the continental US


Results Folder:

All the csv files with results of each model and each absence data generation type.


Report Folder:

LaTeX stuff, main report write-up is Project.tex.

   
