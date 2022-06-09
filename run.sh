#!/bin/bash


# Generate little datasets
python src/generate-datasets.py


# Example for Scatter
python src/service.py \
-i data/generated/globalTemps_scatter.csv -o pru1 \
--title="Maximun Global Temperature Distribution" \
--method=Scatter \
--columns=LandMaxTemperature,LandMinTemperature \
--ylabel="Min Temps" --xlabel="Max Temps"


# Example for ViolinPlot
python src/service.py \
--input=data/generated/tempsInCanadaCameroonAustralia.csv \
--output=scatter_prueba_2 \
--title="Canada, Cameroon and Australia Average Temperature Distribution" \
--method=Violin \
--columns=Country,AverageTemperature \
--ylabel="Average Temperature (in Cº)" --xlabel="Country"



# Example for GeoPlot
python src/service.py \
--input=data/archive/GlobalLandTemperaturesByCity.csv \
--output=citiesMap \
--title=citiesMap \
--method=GeoPlot  \
--columns=Latitude,Longitude


# Example for Decision Tree
python src/service.py \
--ml=DecisionTree \
--input=data/generated/seeds_dataset.csv \
--className=type \
--output=seeds


# Example for Linear Regression
python src/service.py \
--ml=LinearRegression \
--input=data/generated/ml_regression.csv \
--output=maxminTemps


# Example for KMeans
python src/service.py \
--ml=KMeans \
--input=data/generated/ml_kmeans.csv \
--nclusters=3 \
--className=type \
--output=seeds


# Example for Pipeline
python src/service.py \
--ml=Pipeline \
--input=data/generated/seeds_dataset.csv \
--className=type


# Example for Logistic Regression
python src/service.py \
--ml=LogisticRegression \
--input=data/generated/ml_regression.csv \
--output=maxminTemps


# Example for KNeighbors
python src/service.py \
--ml=KNeighbors \
--input=data/generated/ml_logisticRegression.csv \
--output=maxminTemps \
--className=type \
--columns=perimeter,groove_length \
--title="KNeighbors" \
--xlabel="perimeter" \
--ylabel="groove_length"