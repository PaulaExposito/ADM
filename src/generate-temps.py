import pandas as pd


# Major City (January)

majorCity_df = pd.read_csv("data/archive/GlobalLandTemperaturesByMajorCity.csv")

majorCity_df = majorCity_df.dropna()
majorCity_df = majorCity_df[majorCity_df['dt'].str.contains("-01-01")]

print(majorCity_df.describe())
print(majorCity_df.head())

majorCity_df.to_csv('data/generated/majorCityTemps.csv', index = False)



# Major City ML (January)

majorCity_ml_df = pd.read_csv("data/archive/GlobalLandTemperaturesByMajorCity.csv")
majorCity_ml_df = majorCity_ml_df.dropna()
majorCity_ml_df = majorCity_ml_df[majorCity_ml_df['dt'].str.contains("-01-01")]

majorCity_ml_df = majorCity_ml_df[[ "AverageTemperature", "AverageTemperature", "City" ]]

print(majorCity_ml_df.describe())
print(majorCity_ml_df.head())

majorCity_ml_df.to_csv('data/generated/ml_majorCityTemps.csv', index = False)


country_ml_df = pd.read_csv("data/archive/GlobalLandTemperaturesByMajorCity.csv")
country_ml_df = country_ml_df.dropna()
country_ml_df = country_ml_df[country_ml_df['dt'].str.contains("-01-01")]

country_ml_df = country_ml_df[[ "AverageTemperature", "AverageTemperature", "Country" ]]

print(country_ml_df.describe())
print(country_ml_df.head())

country_ml_df.to_csv('data/generated/ml_countryTemps.csv', index = False)




# Global Temps (January)
# Multiple line chart
globalTemps_df = pd.read_csv("data/archive/GlobalTemperatures.csv")

globalTemps_df = globalTemps_df.dropna()

globalTemps_df = globalTemps_df[globalTemps_df.dt.str.contains(r'-01-01')]       # Only January's results
# NUM_DATA = 20

# col_date = globalTemps_df['dt'].tail(NUM_DATA).tolist()
# col_date = list(map(lambda each : each[0:4], col_date))

# col_landMaxTemperature = globalTemps_df['LandMaxTemperature'].tail(NUM_DATA).tolist()
# col_landMinTemperature = globalTemps_df['LandMinTemperature'].tail(NUM_DATA).tolist()
# col_landAverageTemperature = globalTemps_df['LandAverageTemperature'].tail(NUM_DATA).tolist()

print(globalTemps_df.describe())

globalTemps_df.to_csv('data/generated/globalTemps.csv', index = False)


# Global Temps (January) ML
ml_globalTemps_df = pd.read_csv("data/archive/GlobalTemperatures.csv")

ml_globalTemps_df = ml_globalTemps_df.dropna()

ml_globalTemps_df = ml_globalTemps_df[ml_globalTemps_df.dt.str.contains(r'-01-01')]       # Only January's results

ml_globalTemps_df = ml_globalTemps_df[[ "LandMinTemperature", "LandMaxTemperature" ]]

ml_globalTemps_df.to_csv('data/generated/ml_globalTemps.csv', index = False)







