from abc import ABC, abstractmethod         # Module abc (Abstract Base Classes)
from typing import List
from mpl_toolkits.basemap import Basemap, cm

import matplotlib.pyplot as plt
import pandas as pd


# Common interface to all concrete representations 
class Representation(ABC):
    """
    The interface declares common methods to all the concrete representations 
    """

    @abstractmethod
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        pass


"""
Concrete Representation subclasses implement different representations
"""

class LineChart(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        fig = plt.figure()

        for i in range(len(yaxis)):
            plt.plot(xaxis, yaxis[i], label=metadata['ylabel'][i])
        
        plt.title(metadata['title'])
        plt.xlabel(metadata['xlabel'])
        plt.legend()
        fig.autofmt_xdate()
        plt.savefig("output/" + metadata['output'] + "_linechart")


class BarChart(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        fig = plt.figure()

        for i in range(len(yaxis)):
            plt.bar(xaxis, yaxis[i], label=metadata['ylabel'][i])

        plt.title(metadata['title'])
        plt.xlabel(metadata['xlabel'])
        plt.legend()
        fig.autofmt_xdate()
        plt.savefig("output/" + metadata['output'] + "_barchart")


class Histogram(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        fig = plt.figure()
        plt.hist(yaxis)
        plt.title(metadata['title'])
        plt.xlabel(metadata['xlabel'])
        plt.ylabel(metadata['ylabel'][0])
        plt.savefig("output/" + metadata['output'] + "_histogram")


class Scatter(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        fig = plt.figure()

        if 'colors' in metadata:
            plt.scatter(xaxis, yaxis, c=metadata['colors'], cmap='viridis')
        else:
            plt.scatter(xaxis, yaxis, cmap='viridis')

        plt.title(metadata['title'])
        plt.xlabel(metadata['xlabel'])
        plt.ylabel(metadata['ylabel'][0])
        plt.savefig("output/" + metadata['output'] + "_scatter")


class GeoScatter(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        fig = plt.figure(figsize=(12, 8))

        m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180,resolution='c')
        m.drawcoastlines()
        x, y = m(yaxis, xaxis)
        m.scatter(x, y, 10)

        plt.title(metadata['title'])
        plt.savefig("output/" + metadata['output'] + "_geoscatter")



class Context():
    """
    Context is the interface users interact with
    """

    def __init__(self, representation: Representation) -> None:
        self._representation = representation
        self._metadata = None

    @property
    def representation(self) -> Representation:
        return self._representation

    @property
    def metadata(self) -> dict:
        return self._metadata
    
    @representation.setter
    def representation(self, representation: Representation) -> None:
        self._representation = representation

    @metadata.setter
    def metadata(self, metadata: dict) -> None:
        self._metadata = metadata


    def makeChart(self, xaxis = [], yaxis = []) -> None:
        """
        Delegates some work to the strategy object
        """
        self._representation.makeChart(xaxis, yaxis, self._metadata)





# Main

if __name__ == "__main__":

    # Data sources

    df1 = df5 = df7 = pd.read_csv("data/GlobalTemperatures.csv")
    df2 = pd.read_csv("data/GlobalLandTemperaturesByState.csv")
    df3 = df6 = pd.read_csv("data/GlobalLandTemperaturesByCity.csv")
    df4 = pd.read_csv("data/GlobalLandTemperaturesByMajorCity.csv")

    df_seeds = pd.read_csv("data/seeds_dataset.txt", sep=r'\s+', skip_blank_lines=True, skipinitialspace=True)


    # Line chart

    df1 = df1[df1.dt.str.contains(r'-01-01')]       # Only January's results
    NUM_DATA = 20

    col_date = df1['dt'].tail(NUM_DATA).tolist()
    col_date = list(map(lambda each : each[0:4], col_date))

    col_landMaxTemperature = df1['LandMaxTemperature'].tail(NUM_DATA).tolist()
    col_landMinTemperature = df1['LandMinTemperature'].tail(NUM_DATA).tolist()
    col_landAverageTemperature = df1['LandAverageTemperature'].tail(NUM_DATA).tolist()

    columns = [col_landMaxTemperature, col_landAverageTemperature, col_landMinTemperature] 

    metadata1 = {
        "xlabel": "Year",
        "ylabel": [
            "Max. Temp", 
            "Ave. Temp",
            "Min. Temp", 
        ],
        "title": "Global Temperatures at January",
        "output": "plot",
    }

    context1 = Context(LineChart())
    context1._metadata = metadata1
    context1.makeChart(col_date, columns)


    # Bar chart

    df2 = df2.groupby(["Country", "State"])["State"].count().reset_index(name="nentries")
    df2 = df2.groupby("Country")["State"].count().reset_index(name="nstates")

    col_countries = df2['Country'].tolist()
    col_nstates = [ df2['nstates'].tolist() ]

    metadata2 = {
        "xlabel": "Country",
        "ylabel": [
            "Number of states",
        ],
        "title": "Stations per country",
        "output": "plot",
    }

    context2 = Context(BarChart())
    context2._metadata = metadata2
    context2.makeChart(col_countries, col_nstates)


    # Multiple bar chart

    df3 = df3.groupby(["Country", "City"])["City"].count().reset_index(name="nentries")
    df3 = df3.groupby("Country")["City"].count().reset_index(name="ncities")

    df4 = df4.groupby(["Country", "City"])["City"].count().reset_index(name="nentries")
    df4 = df4.groupby("Country")["City"].count().reset_index(name="nmajorcities")
    
    df34 = pd.merge(df3, df4, on="Country")

    # df34 = df34.loc[df34['Country'].isin(['United States', 'Russia', 'South Africa', 'Brazil', 'Pakistan'])]
    df34 = df34.sample(n = 9, random_state = 1)

    col_countries = df34['Country'].tolist()
    col_ncities = df34['ncities'].tolist()
    col_nmajorcities = df34['nmajorcities'].tolist()

    metadata3 = {
        "xlabel": "Country",
        "ylabel": [
            "Number of cities",
            "Number of major cities",
        ],
        "title": "Cities and Major cities per country",
        "output": "plot_multidata",
    }

    context3 = Context(BarChart())
    context3._metadata = metadata3
    context3.makeChart(col_countries, [col_ncities, col_nmajorcities])

    # Histogram

    col_landAverageTemperature = df5['LandAverageTemperature'].tolist()

    metadata5 = {
        "xlabel": "Temperature (in ºC)",
        "ylabel": [
            "Nº measurement", 
        ],
        "title": "Global Temperatures",
        "output": "plot",
    }

    context5 = Context(Histogram())
    context5._metadata = metadata5
    context5.makeChart(yaxis = col_landAverageTemperature)


    # Scatter on map

    col_latitude = df6['Latitude'].tolist()
    col_longitude = df6['Longitude'].tolist()

    for i in range(len(col_latitude)):
        col_latitude[i] = float(col_latitude[i][:-1]) if (col_latitude[i][-1] == 'N') else -float(col_latitude[i][:-1])

    for i in range(len(col_longitude)):
        col_longitude[i] = -float(col_longitude[i][:-1]) if (col_longitude[i][-1] == 'W') else float(col_longitude[i][:-1])

    metadata6 = {
        "title": "Cities in the World",
        "output": "plot",
    }

    context6 = Context(GeoScatter())
    context6._metadata = metadata6
    context6.makeChart(col_latitude, col_longitude)


    Scatter

    df7 = df7.dropna()
    max_temperature = df7['LandMaxTemperature'].tolist()
    min_temperature = df7['LandMinTemperature'].tolist()

    metadata7 = {
        "xlabel": "Max. Temp",
        "ylabel": [
            "Min. Temp",
        ],
        "title": "Maximum Global Temperature Distribution",
        "output": "plot",
    }

    context7 = Context(Scatter())
    context7._metadata = metadata7
    context7.makeChart(max_temperature, min_temperature)


    # Scatter

    col_length = df_seeds['length'].tolist()
    col_width = df_seeds['width'].tolist()
    col_type = df_seeds['type'].tolist()

    metadata_seeds = {
        "xlabel": "Kernel length",
        "ylabel": [
            "Kernel width",
        ],
        "title": "Seeds Distribution",
        "output": "plot-seeds",
        "colors": col_type
    }

    context_seeds = Context(Scatter())
    context_seeds._metadata = metadata_seeds
    context_seeds.makeChart(col_length, col_width)


    print("Exiting...")

