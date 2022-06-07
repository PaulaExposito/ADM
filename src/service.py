from abc import ABC, abstractmethod
from typing import List

import sys, getopt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn import tree
import graphviz


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


class GeoPlot(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        fig = plt.figure(figsize=(12, 8))

        m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180,resolution='c')
        m.drawcoastlines()
        x, y = m(yaxis, xaxis)
        m.scatter(x, y, 10)

        plt.title(metadata['title'])
        plt.savefig("output/" + metadata['output'] + "_geoplot")


class ViolinPlot(Representation):
    def makeChart(self, xaxis = [], yaxis = [], metadata = {}) -> None:
        ax = sns.violinplot(x=xaxis, y=yaxis)
        plt = ax.get_figure()
        # plt.title(metadata['title'])
        # plt.xlabel(metadata['xlabel'])
        # plt.ylabel(metadata['ylabel'][0])
        plt.savefig("output/" + metadata['output'] + "_violinplot")


class ContextRepresentation():
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





class Learning(ABC):
    @abstractmethod
    def learn(self) -> None:
        pass

class Classification(Learning):
    def learn(self) -> None:
        print("Classification")





class Regression(Learning):
    def learn(self) -> None:
        print("Regression")

class Clustering(Learning):
    def learn(self) -> None:
        print("Clustering")

class ContextML():
    def __init__(self, mlAlgorithm: Learning) -> None:
        self._mlAlgorithm = mlAlgorithm
    def learn(self) -> None:
        self._mlAlgorithm.learn()



methods = {
    "LineChart": LineChart(),
    "BarChart": BarChart(),
    "Histogram": Histogram(),
    "Scatter": Scatter(),
    "GeoPlot": GeoPlot(),
    "Violin": ViolinPlot(),
}

mlAlgorithms = {
    "classification" : Classification(),
    "regression" : Regression(),
    "clustering" : Clustering(),
}


def help():
    return """
Usage: python service.py [options]
Options:
    -h,--help                     Help
    -i,--input=<filename>         CSV dataset
    -o,--output=<output>          PNG file with the resulting chart
    -t,--title=<title>            Title of the chart
    -m,--method=<chartname>       Chart to be applied
    -c,--columns=<col1,...,colN>  List with the dataset columns to representate
    --xlabel=<label>              Label for X axis
    --ylabel=<label>              Label for Y axis
    --ml=<classification|
          regression|
          clustering>             Machine learning algorithm
    """

# Main

if __name__ == "__main__":

    inDataset = None
    output = None
    title = None
    chart = None
    columns = None
    xlabel = None
    ylabel = None
    ml = None
    className = None


    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:t:m:c:", ["help", "input=", 
                "output=", "title=", "method=", "columns=", "xlabel=", "ylabel=",
                "ml=","className="])
    except getopt.GetoptError:
        print(help())


    for opt, arg in opts:
        if opt in ("-h, --help"):
            print(help())

        elif opt in ("-i", "--input"):
            inDataset = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-t", "--title"):
            title = arg
        elif opt in ("-m", "--method"):
            chart = arg
        elif opt in ("-c", "--columns"):
            columns = arg.split(",")
        elif opt in ("--xlabel"):
            xlabel = arg
        elif opt in ("--ylabel"):
            ylabel = arg.split(",")
        elif opt in ("--ml"):
            ml = arg
        elif opt in ("--className"):
            className = arg

    
    if (ml != None):
        print(ml)
        print(inDataset)

        try: 
            # Read file and create dataframe
            df = pd.read_csv(inDataset)
            print(df.head())

            # Split the dataset in Train-Test
            X = df.drop(className, axis=1)
            y = df[[className]]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train
            clf_model = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=3, min_samples_leaf=5)   
            clf_model.fit(X_train,y_train)

            # Validation
            y_predict = clf_model.predict(X_test.head())
            print(accuracy_score(y_test.head(), y_predict))


            # Print tree

            target = list(df[className].unique())
            feature_names = list(X.columns)

            # dot_data = tree.export_graphviz(clf_model,
            #                                 out_file=None, 
            #                                 feature_names=feature_names,  
            #                                 class_names=target,  
            #                                 filled=True, rounded=True,  
            #                                 special_characters=True)  
            # graph = graphviz.Source(dot_data)  

            # graph.view()
            # print(graph)

            fig = plt.figure()

            tree.plot_tree(clf_model)

            fig.savefig("output/classification.png")


            # Create context
            mlContext = ContextML(mlAlgorithms.get(ml))



            mlContext.learn()

        except AttributeError:
            print("Algoritmo de machine learning no válido: utilizar classification, regression o clustering")


    if (chart != None and ml == None):
        # Read file and create dataframe
        df = pd.read_csv(inDataset)

        # Create context
        context = ContextRepresentation(methods.get(chart))
        
        # Set metadata
        metadata = {
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": title,
            "output": output
        }
        context.metadata = metadata

        # Create chart
        context.makeChart(df[columns[0]].tolist(), df[columns[1]].tolist())
 
    print("Exiting...")
