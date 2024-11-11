
# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

def exploracion_datos(dataframe):

    """
    Realiza una exploración básica de los datos en el DataFrame dado e imprime varias estadísticas descriptivas.

    Parameters:
    -----------
    dataframe : pandas DataFrame. El DataFrame que se va a explorar.

    Returns:
    --------
    None

    Imprime varias estadísticas descriptivas sobre el DataFrame, incluyendo:
    - El número de filas y columnas en el DataFrame.
    - El número de valores duplicados en el DataFrame.
    - Una tabla que muestra las columnas con valores nulos y sus porcentajes.
    - Las principales estadísticas de las variables numéricas en el DataFrame.
    - Las principales estadísticas de las variables categóricas en el DataFrame.

    """

    print(f"El número de filas es {dataframe.shape[0]} y el número de columnas es {dataframe.shape[1]}")

    print("\n----------\n")

    print(f"En este conjunto de datos tenemos {dataframe.duplicated().sum()} valores duplicados")

    
    print("\n----------\n")


    print("Los columnas con valores nulos y sus porcentajes son: ")
    dataframe_nulos = dataframe.isnull().sum()

    display((dataframe_nulos[dataframe_nulos.values >0] / dataframe.shape[0]) * 100)

    print("\n----------\n")
    print("Las principales estadísticas de las variables númericas son:")
    display(dataframe.describe().T)

    print("\n----------\n")
    print("Las principales estadísticas de las variables categóricas son:")
    display(dataframe.describe(include = "O").T)

    print("\n----------\n")
    print("Las características principales del dataframe son:")
    display(dataframe.info())

class Visualizador:
    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")
    
    def plot_numericas(self, color="grey", tamano_grafica=(15, 5)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        fig, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], color=color, bins=20)
        fig.delaxes(axes[-1])
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout();

    def plot_categoricas(self, color="grey", tamano_grafica=(40, 15)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (40, 10).
        """
        dataframe_cat = self.separar_dataframes()[1]
        _, axes = plt.subplots(2, math.ceil(len(dataframe_cat.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], color=color)
            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.suptitle("Distribución de variables categóricas")

    def plot_relacion(self, vr, tamano_grafica=(40, 12), color="grey"):
        """
        Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.

        Parameters:
            - vr (str): El nombre de la variable en el eje y.
            - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (40, 12).
            - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        Returns:
            No devuelve nada    
        """
        df_numericas = self.separar_dataframes()[0].columns
        meses_ordenados = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in df_numericas:
                sns.scatterplot(x=vr, 
                                y=columna, 
                                data=self.dataframe, 
                                color=color, 
                                ax=axes[indice])
                axes[indice].set_title(columna)
                axes[indice].set(xlabel=None)
            else:
                if columna == "Month":
                    sns.barplot(x=columna, y=vr, data=self.dataframe, order=meses_ordenados, ax=axes[indice],
                                color=color)
                    axes[indice].tick_params(rotation=90)
                    axes[indice].set_title(columna)
                    axes[indice].set(xlabel=None)
                else:
                    sns.barplot(x=columna, y=vr, data=self.dataframe, ax=axes[indice], color=color)
                    axes[indice].tick_params(rotation=90)
                    axes[indice].set_title(columna)
                    axes[indice].set(xlabel=None)

        plt.tight_layout()

    def relacion_numericas(self, variable_dependiente, tamanio=(15,8), paleta="mako"):
        df_numericas= self.separar_dataframes()[0]
        cols_numericas=df_numericas.columns
        nfilas=math.ceil(len(cols_numericas) /2) 
        fig, axes = plt.subplots(nrows= nfilas, ncols= 2, figsize= tamanio)
        axes=axes.flat

        for indice, col in enumerate(cols_numericas):
            if col == variable_dependiente:
                fig.delaxes(axes[indice])
            else:
                sns.scatterplot(x=col ,y=variable_dependiente, data=df_numericas, palette=paleta, ax=axes[indice])

                axes[indice].set_title(f"Relación entre {col} y {variable_dependiente}")
                axes[indice].set_xlabel("")

        fig.delaxes(axes[-1])
        plt.tight_layout()
    
    def analisis_temporal(self, var_respuesta, var_temporal, color = "black", order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):


        """
        Realiza un análisis temporal mensual de una variable de respuesta en relación con una variable temporal. Visualiza un gráfico de líneas que muestra la relación entre la variable de respuesta y la variable temporal (mes), con la línea de la media de la variable de respuesta.


        Params:
        -----------
        dataframe : pandas DataFrame. El DataFrame que contiene los datos.
        var_respuesta : str. El nombre de la columna que contiene la variable de respuesta.
        var_temporal : str. El nombre de la columna que contiene la variable temporal (normalmente el mes).
        order : list, opcional.  El orden de los meses para representar gráficamente. Por defecto, se utiliza el orden estándar de los meses.

        Returns:
        --------
        None

 
        """


        plt.figure(figsize = (15, 5))

        # Convierte la columna "Month" en un tipo de datos categórico con el orden especificado
        self.dataframe[var_temporal] = pd.Categorical(self.dataframe[var_temporal], categories=order, ordered=True)

        # Trama el gráfico
        sns.lineplot(x=var_temporal, 
                     y=var_respuesta, 
                     data=self.dataframe, 
                     color = color)

        # Calcula la media de PageValues
        mean_page_values = self.dataframe[var_respuesta].mean()

        # Agrega la línea de la media
        plt.axhline(mean_page_values, 
                    color='green', 
                    linestyle='--', 
                    label='Media de PageValues')


        # quita los ejes de arriba y de la derecha
        sns.despine()

        # Rotula el eje x
        plt.xlabel("Month");


    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
    

