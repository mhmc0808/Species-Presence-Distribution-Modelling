# Species Distribution Modelling Across the Contiguous US

Predicted species presence across the continental US from citizen science data (iNaturalist), correcting for **observer bias** — the tendency for presence-only data to reflect where humans look, not where a species actually lives.

**Result:** proposed a one-vs-rest pseudo-absence sampling method that significantly outperformed the standard uniform approach across every model tested, reaching a **mean AUC-ROC of 0.932** (Random Forest) vs. 0.854 for the uniform baseline.

Collaborative project with Jackson Cramer, Michael Tiller, and Zac Gavin. Full report found in `Report/` as "Species\_Distribution\_Modelling\_Report.pdf".

\---

## Approach

* Confirmed observer bias statistically (population vs. observation counts, Pearson r = 0.84, p << 0.05) before deciding how to handle it
* Proposed a **one-vs-rest pseudo-absence sampler** as an alternative to random uniform sampling, to counteract that bias
* Trained Logistic Regression, Random Forest, and GBDT on both absence methods, tuned via grid search
* Validated the improvement with Wilcoxon signed-rank tests (significant at 5% level across all 6 model/tuning combinations)

\---

## Repository

```
SRC/     - data pipeline, model training, significance testing, prediction maps
Data/     - species observations + environmental features (NCEI, USDA, NACIS, SimpleMaps)
Results/  - per-model evaluation outputs
Report/   - Full report, plots and references

```



SRC:



1\. `dataloadingfunction.py`

&#x20;  - Subsets our data to only continental US.

&#x20;  - Loads all additional data gathered

&#x20;  - Handles all data preprocessing and feature engineering, merging data to latitude and longitude in either train or test set.

&#x20;  - Attaches environmental features to species data by latitude/longitude coordinates using KD-tree nearest-neighbour matching.

&#x20;  - Calculates each species data distance to nearest coastline, proportion of farmland by county.



2\. `human\_pop\_initial\_eda.py`

&#x20;  - training/test Data Info

&#x20;  - model plausibility

&#x20;  - human population analysis by comparing frequency of species data to US city population data.

&#x20;    

3\. `modelFitting.py`

&#x20;  - loads full dataset from `dataloadingfunction.py`, also requires data filecb\_2018\_us\_state\_500k.zip.

&#x20;  - generates pseudo-absence data by one-vs-rest or uniform random method.

&#x20;  - for each species with more than 1000 observations, modelfitting:

&#x20;    - builds training data

&#x20;    - trains model using various ML models

&#x20;    - performs GridSearch for hyperparameter tuning

&#x20;    - evaluates model performance on test set (using AUC-ROC, accuracy, log-loss).

&#x20;  - all results are saved to csv files stored in `Results` folder, where the name specifies the pseudo-absence data collection method and ML model used.

&#x20;  - to change the model and the type of training data (one vs rest and Psuedabsence), we must comment and uncomment the appropriate lines of code. 



4\. `SDM.py`

&#x20;  - produces prediction maps across the US for different species, indicating where our models determine these species are likely to be present.

&#x20; 

5\. `Analysis.py`

&#x20;  - conducts hypothesis tests

&#x20;  - both for difference between AUC-ROC of models trained on uniform and one-vs-rest pseudo-absences

&#x20;  - and for difference between  tuned and untuned models.

&#x20;    

Data: 



1\. `species\_train.npz`

&#x20;  - contains training dataset (presence-only), where

&#x20;    - train\_locs - has \[latitude, longitude] of each observation

&#x20;    - train\_ids - species ID for each observation

&#x20;    - taxon\_ids - list of unique species IDs present

&#x20;    - taxon\_names - species names corresponding to taxon\_ids



2\. `species\_test.npz`

&#x20;  - contains testing dataset (presence and absence data), where

&#x20;    - test\_locs - has \[latitude, longitude] coordinates of where predictions are made

&#x20;    - taxon\_ids - list of unique species IDs included in test set

&#x20;    - taxon\_names - species names corresponding to taxon\_ids

&#x20;    - test\_pos\_inds - list of indices in test\_locs for each species determining where the species is present. All other locations for that species are considered absent.

&#x20;   

3\. `farmland\_proportion\_by\_county.csv`

&#x20;  - contains details on farmland for all US counties.

&#x20;  - in our analysis, we only extract State, County, Value (state and county to match with our data, value tells us the proportion of the given county that is farmland)

&#x20;  - taken from USDA National Agricultural Statistics Service website: https://quickstats.nass.usda.gov/results/7A20CA20-5831-3B08-B570-788F9B3157BE

&#x20;  - In order to extract this dataset, go to the USDA Quick Stats website: https://quickstats.nass.usda.gov/ and choose the following options: Group: FARMS \& LANDS \& ASSETS, Commodity: FARM OPERATIONS, Data Item: FARM OPERATIONS - AREA OPERATED, MEASURED IN PCT OF TOTAL LAND. Geographic Level: COUNTY, then click Get Data.

&#x20; 

4\. `nclimgrid\_prcp.nc`

&#x20;  - 2020 Monthly Averages precipitation (mm)

&#x20;  - Source: National Centers for Environmental Information (NCEI)

&#x20;  - Used to extract mean and variance precipitation grid data ranging across the continental US

&#x20;    

5\. `nclimgrid\_tavg.nc`

&#x20;  - 2020 Monthly Averages temperature (degrees celsius)

&#x20;  - Source: National Centers for Environmental Information (NCEI)

&#x20;  - Used to extract mean and variance temperature grid data ranging across the continental US

&#x20;    

6\. `simplemaps\_uscities\_basicv1.91.zip`

&#x20;   -Data on population density of US cities, used for human bias analysis



7\. `soilw.0-10cm.gauss.2020.nc`

&#x20;  - 2020 4x Daily Volumetric Soil Moisture 10cm below ground

&#x20;  - Source: National Centers for Environmental Information (NCEI)

&#x20;  - Used to extract mean and variance soil moisture data ranging across the continental US

&#x20;    

8\. `uscities.csv`

&#x20;   -Data on population density of US cities, used for human bias analysis. Extracted from `simplemaps\_uscities\_basicv1.91.zip` folder.



9.. `uwnd.10m.gauss.2020.nc`

&#x20;  - 2020 10m above ground (m/s) average

&#x20;  - Source: National Centers for Environmental Information (NCEI)

&#x20;  - Used to extract mean wind speed and mean direction grid data ranging across the continental US



10\. `VIIRS-Land\_v001\_NPP13C1\_S-NPP\_20200101\_c20240126161431.nc`

&#x20;  - 2020 Daily Averages, derived from the NOAA Climate Data Record (CDR)

&#x20;  - Source: National Centers for Environmental Information (NCEI)

&#x20;  - Used to extract NDVI grid data ranging across the continental US

&#x20;    

11\. `vwnd.10m.gauss.2020.nc`

&#x20;  - 2020 10m above ground (m/s) average

&#x20;  - Source: National Centers for Environmental Information (NCEI)

&#x20;  - Used to extract mean wind speed and mean direction grid data ranging across the continental US

\---

### Python Libraries Required

`numpy`, `pandas`, `zipfile`, `matplotlib`, `seaborn`, `reverse\_geocoder`, `cartopy`, `scipy`, `sklearn`, `geopandas`,`seaborn`, `rasterio`, `shapely`, `random`, `netCDF4`, `geodatasets`, `os`, `wget`, `IPython`, `graphviz`

\---

**Generative AI use:** used to assist with loading/formatting external data; not used for code generation or report writing.

