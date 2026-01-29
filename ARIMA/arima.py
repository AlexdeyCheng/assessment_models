import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings

warnings.filterwarnings("ignore")

df = pd.read_excel(
    "D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/2023_MCM_Problem_C_Data.xlsx",
    header=1
)

print(df.isnull().sum())