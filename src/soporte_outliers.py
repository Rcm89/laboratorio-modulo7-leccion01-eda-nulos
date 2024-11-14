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
import plotly.express as px



# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon

# Para generar combinaciones de listas
# -----------------------------------------------------------------------
from itertools import product, combinations

# Gestionar warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

def describe_outliers(dataframe: pd.DataFrame, k=1.5, ordenados = True):

    diccionario_outliers = []
   
    columnas_numericas = dataframe.select_dtypes(np.number).columns

    for columna in columnas_numericas:
        Q1, Q3 = np.nanpercentile(dataframe[columna], (25,75))
        IQR = Q3 - Q1

        limite_inferior = Q1 - (IQR * k)
        limite_superior = Q3 + (IQR * k)

        condicion_inferior = dataframe[columna] < limite_inferior
        condicion_superior = dataframe[columna] > limite_superior

        df_outliers = dataframe[condicion_inferior | condicion_superior]
        
        diccionario_outliers.append({
            'columna': columna,
            'n_outliers': df_outliers.shape[0],
            'limite_inf': limite_inferior,
            'limite_sup': limite_superior,
            '%_outliers': round((df_outliers.shape[0] / dataframe.shape[0]) * 100, 2)
        })

    resultado = pd.DataFrame(diccionario_outliers).sort_values(by='n_outliers', ascending=False) if ordenados == True else pd.DataFrame(diccionario_outliers)
        
    display(resultado)

class GestionOutliersMultivariados:

    def __init__(self, dataframe, contaminacion = [0.01, 0.05, 0.1, 0.15]):
        self.dataframe = dataframe
        self.contaminacion = contaminacion

    def separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")


    def visualizar_outliers_bivariados(self, vr, tamano_grafica = (20, 15)):

        num_cols = len(self.separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.separar_variables_tipo()[0].columns):
            if columna == vr:
                fig.delaxes(axes[indice])
        
            else:
                sns.scatterplot(x = vr, 
                                y = columna, 
                                data = self.dataframe,
                                ax = axes[indice])
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None, ylabel = None)
        fig.delaxes(axes[-1])
        plt.tight_layout()


    
    