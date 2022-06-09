#!/bin/bash


##############################################################################
###  PLOTS
##############################################################################


# Multiple line chart
python src/service.py \
--input=data/generated/globalTemps.csv \
--output=out \
--columns=dt,LandAverageTemperature,LandMaxTemperature,LandMinTemperature \
--method=LineChart \
--title="Evolución de las temperaturas a 1 de enero" \
--xlabel=Fecha \
--ylabel="Avg. Temp, Max. Temp, Min. Temp"


# Multiple bar chart
python src/service.py \
--input=data/generated/tempsByCity_multipleBarChart.csv \
--output=out \
--columns=Country,ncities,nmajorcities \
--method=BarChart \
--title="Cities and MajorCities por país" \
--xlabel=Country \
--ylabel="Nº of cities, Nº of major cities"


# Histogram
python src/service.py \
--input=data/generated/globalTemps.csv \
--output=out \
--columns=dt,LandAverageTemperature \
--method=Histogram \
--title="Distribución de la temperatura" \
--xlabel="Temperature (ºC)" \
--ylabel="Nº of measurements"


# Scatter
python src/service.py \
--input=data/generated/globalTemps.csv \
--output=out \
--title="Maximun Global Temperature Distribution in January" \
--method=Scatter \
--columns=LandMaxTemperature,LandMinTemperature \
--ylabel="Min Temps" \
--xlabel="Max Temps"


# Violin Plot
python src/service.py \
--input=data/generated/majorCityTemps.csv \
--output=out \
--title="Distribucion de las temperaturas en ciudades importantes" \
--method=Violin \
--columns=Country,AverageTemperature \
--xlabel=Country \
--ylabel="Temp. Media (ºC)" 




##############################################################################
###  MACHINE LEARNING
##############################################################################


# Pipeline
python src/service.py \
--input=data/generated/ml_majorCityTemps.csv \
--ml=Pipeline \
--output=out \
--className=City

python src/service.py \
--input=data/generated/ml_countryTemps.csv \
--ml=Pipeline \
--output=out \
--className=Country


# Decision Tree
python src/service.py \
--input=data/generated/ml_countryTemps.csv \
--ml=DecisionTree \
--output=out \
--className=Country


# Linear Regression
python src/service.py \
--input=data/generated/ml_globalTemps.csv \
--ml=LinearRegression \
--output=out \
--title="Linear regression of global temperatures in January" \
--xlabel="min temps" \
--ylabel="max temps"

# Logistic Regression
python src/service.py \
--input=data/generated/ml_countryTemps.csv \
--ml=LogisticRegression \
--output=out \
--title="Logistic regression of global temperatures in January" \
--xlabel="min temps" \
--ylabel="max temps" \
--className="Country"


# KMeans
python src/service.py \
--ml=KMeans \
--input=data/generated/ml_globalTemps.csv \
--nclusters=3 \
--output=out \
--className=LandMaxTemperature \
--xlabel="Min Temps" \
--ylabel="Max Temps" \
--title="Agrupamiento de temperaturas globales (de enero) para k=3"


