import engine

import pandas as pd

import numpy as np

import time

gvns = engine.GVNS()


AMPARO = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Amparo.xlsx")


data = np.array(gvns.perform_GVNS_during(AMPARO.matrix, 30*60, 3))

df = pd.DataFrame(data.transpose(), columns=["Value", "Time"])

df.to_csv("amparo_30mins_k3.csv", index=False, na_rep="N/A")