# --------------------
# MODULES
import numpy as np
import pandas as pd
import seaborn as sbs
# --------------------

class MDP:
    #File with data and m="How many items have to be chosen"
    def __init__(self, file, m=25):
        
        self.m = m

        #Get data
        excel = pd.read_excel(file, header = None)
        self.dist_mat = np.array(excel)

        #Arrange items
        self.items_array = [np.empty(len(self.dist_mat[0]), dtype = float), np.ones(len(self.dist_mat[0]), dtype=int)
        # First row -> contributions
        # Second row -> In solution?
        #   CODE -> 1:Not in solution -- 0:In solution
           
 
class GRASP():
                            
    
                


        
    
        
        
        
    