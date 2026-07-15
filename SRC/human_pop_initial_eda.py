import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import reverse_geocoder as rg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import zipfile
from scipy.spatial import cKDTree
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


# required data
# 1. species_train.npz
# 2. simplemaps_uscities_basicv1.91.zip (uscities.csv)
# 3. species_test.npz

# loading training data    
data = np.load('species_train.npz', allow_pickle=True)
train_locs = data['train_locs']  
train_ids = data['train_ids']
                                
species = data['taxon_ids']
species_names = dict(zip(data['taxon_ids'], data['taxon_names']))  # latin names of species 

# put into pandas dataframe
data = pd.DataFrame(train_locs, columns=['latitude', 'longitude'])
data['species_id'] = train_ids
data['species_name'] = data['species_id'].map(species_names)
print(data.head())


# create country  and other geographic columns
coords = list(zip(data['latitude'], data['longitude']))
results = rg.search(coords)
data['country_code'] = [r['cc'] for r in results]
data['city'] = [r['name'] for r in results]
data['region'] = [r['admin1'] for r in results]
data['county'] = [r['admin2'] for r in results]
data.head()


#------------------------------------------------------------------------------

# refine data to include only US species

# disclude all data not within the US

data = data[data["country_code"] == "US"]

# remove country_code column

data = data.drop(['country_code'], axis=1)
data.rename(columns={'region': 'state'}, inplace=True)


# refine to only include continental US

continental_states = [
    "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts",
    "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
    "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "Washington DC"
]

data = data[data["state"].isin(continental_states)]

data["species_id"].value_counts()
len(data)



#------------------------------------------------------------------------------

# plot distribution of species 41301 (brasilian free tailed bat)


species_data = data[data['species_id'] == 41301]

fig = plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.LambertConformal())
ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.STATES, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE)

# Plot points
ax.scatter(
    species_data['longitude'], 
    species_data['latitude'], 
    color='red', s=12,
    transform=ccrs.PlateCarree(),
    label='Species 41301'
)

plt.title("Locations of Species 41301 in the Continental US")
plt.legend()
plt.show()

#------------------------------------------------------------------------------


# carry out correlation test with species and human populations by city

# load in population data
with zipfile.ZipFile('simplemaps_uscities_basicv1.91.zip') as z:
    with z.open('uscities.csv') as f:
        human_pop = pd.read_csv(f)[['lat','lng','city', 'state_name', 'population']]
human_pop = human_pop.rename(columns={'state_name': 'state', 'lat':'latitude','lng':'longitude'})


# build a KDTree for nearest-neighbor matching
human_coords = np.column_stack((human_pop['latitude'], human_pop['longitude']))
tree = cKDTree(human_coords)

# get nearest neighbor for each observation
data_coords = np.column_stack((data['latitude'], data['longitude']))
_, indices = tree.query(data_coords)

# add population from nearest human_pop entry
data['population'] = human_pop.iloc[indices]['population'].values

# rename column
data = data.rename(columns={'population': 'human_population'})

# bin the latitudes and longitudes to 1 degree x 1 degree
data['lat_bin'] = data['latitude'].round()
data['lon_bin'] = data['longitude'].round()

# get the species counts for the bins
counts = (data.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count'))
counts = counts[counts['count'] > 0] # remove zero counts to avoid error (log(0))
counts_log = np.log10(counts["count"]) # take log to normalize data
counts["log_count"] = counts_log

# get human population by binned area
pop_area = data.groupby(['lat_bin', 'lon_bin'], as_index=False)['human_population'].sum()
pop_area = pop_area[pop_area['human_population'] > 0]
pop_area['log_pop'] = np.log10(pop_area['human_population'])


# merge species counts and human population by lat/lon bins
merged = counts.merge(pop_area, on=['lat_bin', 'lon_bin'], how='inner')

# plot counts and regression line
sns.regplot(
    x=merged['log_pop'],
    y=merged['log_count'],
    line_kws={'color': 'red'}
)
plt.title("log10(Human Population) vs log10(Species Counts)")
plt.xlabel("log10(Human Population)")
plt.ylabel("log10(Species Counts)")
plt.show()


# calculate Pearson correlation
pearson_corr, p_value = pearsonr(merged['log_count'], merged['log_pop'])
print(f"Pearson correlation (log-transformed totals per bin): {pearson_corr:.4f}")
print(f"P-value: {p_value:.4e}")

# fit the model to look at residuals
X = merged['log_pop'].values.reshape(-1,1)
y = merged['log_count'].values

model = LinearRegression()
model.fit(X, y)

fitted = model.predict(X)
residuals = y - fitted

# residual vs. fitted plot
plt.figure(figsize=(6,5))
plt.scatter(fitted, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.show()

# qq plot
plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Residuals Quantiles")
plt.show()


#------------------------------------------------------------------------------


# loading test data 
data_test = np.load('species_test.npz', allow_pickle=True)
test_locs = data_test['test_locs']    
test_pos_inds = dict(zip(data_test['taxon_ids'], data_test['test_pos_inds']))    

test_locs = data_test['test_locs']
taxon_ids = data_test['taxon_ids']
taxon_names = data_test['taxon_names']
test_pos_inds = dict(zip(taxon_ids, data_test['test_pos_inds']))

# turn test data into a more usable pandas dataframe
df_list = []
for taxon_id, inds in test_pos_inds.items():
    name = taxon_names[np.where(taxon_ids == taxon_id)][0]
    locs = test_locs[inds]
    df_tmp = pd.DataFrame(locs, columns=['latitude', 'longitude'])
    df_tmp['taxon_id'] = taxon_id
    df_tmp['taxon_name'] = name
    df_list.append(df_tmp)
species_df = pd.concat(df_list, ignore_index=True)

# add state column to test data
coords = list(zip(species_df['latitude'], species_df['longitude']))
unique_coords = list(dict.fromkeys(coords))
results_unique = rg.search(unique_coords, mode=1)   # mode=1 → one result per coord
coord_to_region = {
    coord: result['admin1']
    for coord, result in zip(unique_coords, results_unique)
}
species_df['state'] = [coord_to_region[c] for c in coords]


# refine to only include continental states
species_df = species_df[species_df["state"].isin(continental_states)]


species_df["taxon_id"].value_counts()
len(species_df)


