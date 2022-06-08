import pandas as pd
import os


# Data sources

df1 = df5 = df7 = pd.read_csv("data/archive/GlobalTemperatures.csv")
# df2 = pd.read_csv("data/archive/GlobalLandTemperaturesByState.csv")
# df3 = df6 = pd.read_csv("data/archive/GlobalLandTemperaturesByCity.csv")
# df4 = pd.read_csv("data/archive/GlobalLandTemperaturesByMajorCity.csv")
# df_violin = pd.read_csv("data/archive/GlobalLandTemperaturesByCountry.csv")



os.makedirs('data/generated', exist_ok=True)


# # Multiple bar chart

# df3 = df3.groupby(["Country", "City"])["City"].count().reset_index(name="nentries")
# df3 = df3.groupby("Country")["City"].count().reset_index(name="ncities")

# df4 = df4.groupby(["Country", "City"])["City"].count().reset_index(name="nentries")
# df4 = df4.groupby("Country")["City"].count().reset_index(name="nmajorcities")

# df34 = pd.merge(df3, df4, on="Country")

# # df34 = df34.loc[df34['Country'].isin(['United States', 'Russia', 'South Africa', 'Brazil', 'Pakistan'])]
# df34 = df34.sample(n = 9, random_state = 1)

# # col_countries = df34['Country'].tolist()
# # col_ncities = df34['ncities'].tolist()
# # col_nmajorcities = df34['nmajorcities'].tolist()

# df34.to_csv('data/generated/tempsByCity_multipleBarChart.csv', index=False)



# # Scatter

# df7 = df7.dropna()
# df7.to_csv('data/generated/globalTemps_scatter.csv', index=False)


# # Violin

# df_violin = df_violin.dropna()
# df_violin = df_violin.loc[df_violin['Country'].isin(["Canada", "Cameroon", "Australia"])]
# df_violin.to_csv('data/generated/tempsInCanadaCameroonAustralia.csv', index=False)


# ML Classification

# df6 = df6.dropna()
# df6 = df6.drop(columns=['dt', 'Country', 'Latitude', 'Longitude'])
# df6 = df6[['AverageTemperature', 'AverageTemperatureUncertainty', 'City']]
# df6.to_csv('data/generated/ml_classification.csv', index = False)


df_seeds = df_seeds_dt = pd.read_csv("data/seeds_dataset.txt", sep = r'\s+', skip_blank_lines = True, skipinitialspace = True)


# ML Classification

df_seeds_dt['type'] = df_seeds_dt['type'].astype(str)
df_seeds_dt['type'] = df_seeds_dt['type'].replace(['1', '2', '3'], ['Kama', 'Rosa', 'Canadian'])
df_seeds_dt.to_csv('data/generated/seeds_dataset.csv', index = False)


# ML Linear Regression

df5 = df5.dropna()
df5 = df5[['LandMaxTemperature', 'LandMinTemperature']]
df5.to_csv('data/generated/ml_regression.csv', index = False)


# ML Logistic Regression
df_seeds_lr = df_seeds_kmeans = pd.read_csv("data/seeds_dataset.txt", sep = r'\s+', skip_blank_lines = True, skipinitialspace = True)
df_seeds_kmeans = df_seeds_kmeans[['area', 'perimeter', 'type']]
df_seeds_lr.to_csv('data/generated/ml_logisticRegression.csv', index = False)


# ML Clustering

df_seeds_kmeans = df_seeds_kmeans[['area', 'perimeter', 'type']]
df_seeds_kmeans.to_csv('data/generated/ml_kmeans.csv', index = False)