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

