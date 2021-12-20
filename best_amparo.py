import engine

import pandas as pd

import numpy as np

grasp = engine.GRASP()

AMPARO = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Amparo.xlsx")

alphas = np.array([0, 0.2, 0.3])

primera = True
for alpha in alphas:

    paquete = np.array(grasp.perform_during(AMPARO.matrix, alpha, 30*60))
    columna1 = f"{alpha}"
    columna2 = "Time" + f"{alpha}"

    if primera:
        DF = pd.DataFrame(paquete.transpose(),
                          columns=[columna1, columna2])
        primera = False
    else:
        DF = pd.concat(
            [DF, pd.DataFrame(paquete.transpose(),
                              columns=[columna1, columna2])], axis=1)


DF.to_csv("AMPARO_comparing_low_alphas.csv", index=False, na_rep="N/A")
