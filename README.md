# Análisis de datos masivos

## Setup

```bash
pip3 install virtualenv
cd /home/user/path/to/here
python3 -m virtualenv '.env'
source '.env/bin/activate'
pip3 install -r requirements.txt
deactivate
```

Descargar y guardar en ```/data``` los datasets de [Climate Change: Earth Surface Temperature Data](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data).


## Usage example

```
python src/service.py -i data/generated/globalTemps_scatter.csv -o pru1 -t prueba -m Scatter -c LandMaxTemperature,LandMinTemperature --ylabel="Min Temps" --xlabel="Max Temps"
```





