import pandas as pd

# Read the dataset
df = pd.read_csv("data/archive/GlobalLandTemperaturesByMajorCity.csv")

# Get first five rows
# print(df.head())

# Get columns
# print(df.columns)

# Drop columns
# df = df.drop(columns=['LandAndOceanAverageTemperature'])

# Length of the dataframe
# print(len(df))

# Subdataset
# print(df.iloc[:10, 5:7])      # Get the first 10 rows and index 5-7 columns (row 10 and column 7 not included)
# print(df.loc[[2, 3, 7], ['dt', 'LandAndOceanAverageTemperature']])      # Specify rows and colums

# Datatypes
# print(df.dtypes)

# Describe the dataset
print(df.describe())

# Unique values
# print(df.LandAverageTemperature.unique())
# print(df.LandAverageTemperature.nunique())      # Number of unique values

# Sampling
# print(df.sample(frac = 0.25))       # Take the 25% of the data

# Number of null values
# print(df.isnull().sum())

# Rename column
# df.rename(columns = { "OldName": "NewName" })

# Group by
# df.groupby("ColunmName")["AnotherColumn"].sum()

# Correlation matrix
# print(df.corr(method='pearson'))

# n largest and smallest values
# print(df.nlargest(6, "LandMaxTemperature"))
# print(df.nsmallest(6, "LandMaxTemperature"))

# Get information about the dataframe
# print(df.info())

