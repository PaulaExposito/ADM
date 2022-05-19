import pandas as pd
import os


# Data sources

df1 = df5 = df7 = pd.read_csv("data/archive/GlobalTemperatures.csv")
df2 = pd.read_csv("data/archive/GlobalLandTemperaturesByState.csv")
df3 = df6 = pd.read_csv("data/archive/GlobalLandTemperaturesByCity.csv")
df4 = pd.read_csv("data/archive/GlobalLandTemperaturesByMajorCity.csv")
df_violin = pd.read_csv("data/archive/GlobalLandTemperaturesByCountry.csv")

# df_seeds = pd.read_csv("data/seeds_dataset.txt", sep=r'\s+', skip_blank_lines=True, skipinitialspace=True)


os.makedirs('data/generated', exist_ok=True)


# Multiple bar chart

df3 = df3.groupby(["Country", "City"])["City"].count().reset_index(name="nentries")
df3 = df3.groupby("Country")["City"].count().reset_index(name="ncities")

df4 = df4.groupby(["Country", "City"])["City"].count().reset_index(name="nentries")
df4 = df4.groupby("Country")["City"].count().reset_index(name="nmajorcities")

df34 = pd.merge(df3, df4, on="Country")

# df34 = df34.loc[df34['Country'].isin(['United States', 'Russia', 'South Africa', 'Brazil', 'Pakistan'])]
df34 = df34.sample(n = 9, random_state = 1)

# col_countries = df34['Country'].tolist()
# col_ncities = df34['ncities'].tolist()
# col_nmajorcities = df34['nmajorcities'].tolist()

df34.to_csv('data/generated/tempsByCity_multipleBarChart.csv', index=False)



# Scatter

df7 = df7.dropna()
df7.to_csv('data/generated/globalTemps_scatter.csv', index=False)


# Violin

df_violin = df_violin.dropna()
df_violin = df_violin.loc[df_violin['Country'].isin(["Canada", "Cameroon", "Australia"])]
df_violin.to_csv('data/generated/tempsInCanadaCameroonAustralia.csv', index=False)
