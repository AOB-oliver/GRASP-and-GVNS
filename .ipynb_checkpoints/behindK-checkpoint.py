import numpy as np
import pandas as pd
import engine
import time

# In order to have an approach of how much time rests


def Time_to_finish(t):
    total = 8*5*60

    current_performing = time.time()-t

    rests = int(total - current_performing)

    hours = int(rests/3600)
    minutes = int((rests % 3600) / 60)
    sec = int((rests % 3600) % 60)

    return hours, minutes, sec


# Import instances and load the GRASP engine
gvns = engine.GVNS()

AMPARO = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Amparo.xlsx")
BORJA = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Borja.xlsx")
DANIEL = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Daniel.xlsx")
EMILIO = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Emilio.xlsx")
JOSE = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Jose.xlsx")
MARIAJESUS = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/MariaJesus.xlsx")
RAQUEL = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Raquel.xlsx")
VIRGINIA = engine.Instance(
    "/home/adrian/UNIVERSIDAD/4to_Curso" +
    "/A.MIO/PROJECT/pythoncode/MDP_instances/Virginia.xlsx")

# Array with considered K's
Ks = np.arange(1, 6)

# Lists with instances and names
instancias = [AMPARO, BORJA, DANIEL,
              EMILIO, JOSE, MARIAJESUS, RAQUEL, VIRGINIA]
nombres = ["AMPARO", "BORJA", "DANIEL", "EMILIO",
           "JOSE", "MARIAJESUS", "RAQUEL", "VIRGINIA"]

# Main loop with the perform of GRASP over the instances.
i = 0
t_0 = time.time()
primera = True
for instancia in instancias:
    for K in Ks:

        paquete = np.array(gvns.perform_GVNS_during(instancia.matrix, 1*60, K))
        columna1 = nombres[i] + f"{K}"
        columna2 = "Time" + nombres[i] + f"{K}"

        if primera:
            DF = pd.DataFrame(paquete.transpose(),
                              columns=[columna1, columna2])
            primera = False
        else:
            DF = pd.concat(
                [DF, pd.DataFrame(paquete.transpose(),
                                  columns=[columna1, columna2])], axis=1)

        hours, minutes, secs = Time_to_finish(t_0)
        print(f"Quedan {hours}h {minutes}m {secs}s para terminar.")

    i += 1

# Save the data in a csv-file
DF.to_csv("behindKs.csv", index=False, na_rep="N/A")
