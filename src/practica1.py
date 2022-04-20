from abc import ABC, abstractmethod         # Module abc (Abstract Base Classes)
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# Common interface to all concrete representations 
class Representation(ABC):
    """
    The interface declares common methods to all the concrete representations 
    """

    @abstractmethod
    def makeChart(self, xaxis : List, yaxis : List, metadata: dict) -> None:
        pass


"""
Concrete Representation subclasses implement different representations
"""

class LineChart(Representation):
    def makeChart(self, xaxis : List, yaxis : List, metadata: dict) -> None:
        fig = plt.figure()
        plt.plot(xaxis, yaxis)
        plt.title(metadata['title'])
        plt.xlabel(metadata['xlabel'])
        plt.ylabel(metadata['ylabel'])
        fig.autofmt_xdate()
        plt.savefig("output/" + metadata['output'] + "_linechart")


class BarChart(Representation):
    def makeChart(self, xaxis : List, yaxis : List, metadata: dict) -> None:
        fig = plt.figure()
        plt.bar(xaxis, yaxis)
        plt.title(metadata['title'])
        plt.xlabel(metadata['xlabel'])
        plt.ylabel(metadata['ylabel'])
        plt.savefig("output/" + metadata['output'] + "_barchart")


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


    def makeChart(self, xaxis : List, yaxis : List) -> None:
        """
        Delegates some work to the strategy object
        """
        self._representation.makeChart(xaxis, yaxis, self._metadata)





# Main

if __name__ == "__main__":

    # Data sources

    df1 = pd.read_csv("data/GlobalTemperatures.csv")
    df2 = pd.read_csv("data/GlobalLandTemperaturesByState.csv")

    
    # Line chart

    col_date = df1['dt'].tail(50).tolist()
    col_landAverageTemperature = df1['LandMaxTemperature'].tail(50).tolist()

    metadata1 = {
        "xlabel": "LandMaxTemperature",
        "ylabel": "Date",
        "title": "Global Temperatures",
        "output": "plot",
    }

    context1 = Context(LineChart())
    context1._metadata = metadata1
    context1.makeChart(col_date, col_landAverageTemperature)


    # Bar chart

    df2 = df2.groupby(["Country", "State"])["State"].count().reset_index(name="nentries")
    df2 = df2.groupby("Country")["State"].count().reset_index(name="nstates")

    col_countries = df2['Country'].tolist()
    col_nstates = df2['nstates'].tolist()

    metadata2 = {
        "xlabel": "Country",
        "ylabel": "Number of states",
        "title": "Stations per country",
        "output": "plot",
    }

    context2 = Context(BarChart())
    context2._metadata = metadata2
    context2.makeChart(col_countries, col_nstates)


    print("Exiting...")

