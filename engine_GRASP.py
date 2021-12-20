## Engine GRASP -- A revisitation ##


# Modules

import time
import numpy as np
import pandas as pd


class Instance:

    def __init__(self, file_path):
        self.name = file_path[:-5]
        self.matrix = np.array(pd.read_excel(file_path, header=None))
        self.size = len(self.matrix[0])


class GRASP:

    # Not in use
    def locate_max(self, mat):
        """
        Returns the first couple of indices satisfaying mat has a max in that
        position
        """
        indices = list(zip(np.where(mat == np.amax(mat))[0][:],
                           np.where(mat == np.amax(mat))[1][:]))
        candidates = np.arange(len(indices))
        return indices[np.random.choice(candidates)]
##

    def initialize_sol_set(self, instance):
        """
        Will return a random element such that the distance bwt them is maximal
        """

        elements = np.arange(len(instance[0]))

        return list(np.random.choice(elements, 1))

# We need another contribution-creator function:

    def create_contributions(self, instance, sol_set):
        """
        Will return an array with the contributions of each item to the
        current sol_set. The contribution of the items already in sol_set is
        stated zero.
        """
        length = len(instance[0])
        contributions = np.zeros(length)
        for i in range(length):
            if i in sol_set:
                pass
            else:
                contributions[i] = instance[i, sol_set[0]]

        return contributions

    def update_contributions(self, instance, contributions, sol_set):
        """
        Updates the current contributions array depending on the new item
        in sol_set
        """
        length = len(instance[0])
        for i in range(length):
            if i in sol_set:
                pass
            else:
                contributions[i] += instance[i][sol_set[-1]]
        # Contribution of the new item setted to 0
        contributions[sol_set[-1]] = 0

        return contributions  # Returns new contributions array

    # Let's construct the Restricted Cadidate List
    def build_RCL(self, contributions, alpha):
        """
        Will return a list with items in the RCL
        """
        _max = contributions.max()
        _min = contributions.min()
        gte = _max - alpha*(_max - _min)

        RCL = np.where(contributions >= gte)[0][:]

        return RCL

    # Random pick from an RCL
    def pick_item(self, RCL):
        return np.random.choice(RCL)

    def construct_solution(self, instance, alpha, size=25):
        """
        This method implements the construction phase of an iteration of GRASP
        """
        # Initialize the solution set
        sol_set = self.initialize_sol_set(instance)
        obj_func = 0

        contributions = self.create_contributions(instance, sol_set)

        while len(sol_set) < size:

            RCL = self.build_RCL(contributions, alpha)
            sol_set.append(self.pick_item(RCL))
            obj_func += contributions[sol_set[-1]]
            contributions = self.update_contributions(instance,
                                                      contributions,
                                                      sol_set)

        return sol_set, obj_func, contributions  # We want the final
        # contribution to speed up future calculations

    def exchange_component_search(self, instance, sol_set, index,
                                  contributions, obj_func):
        """
        Will perform a local search based on an exchange of one item in the
        solution by an item not in the current solution
        """
        out_value = np.sum([instance[sol_set[index], x] for x in sol_set])
        in_values = contributions - instance[:, sol_set[index]]

        max_in = in_values.max()

        if max_in > out_value:
            loc = np.where(in_values == in_values.max())[0][0]

            # Update obj_function
            obj_func = obj_func - out_value + in_values[loc]

            # Exchange items in sol_set
            sol_set[index] = loc

            # Update contributions
            contributions = in_values + instance[:, loc]

            # Set the contributions of the new item to 0
            contributions[loc] = 0

            # Return True if an exchange has been succesfully performed
            return sol_set, obj_func, contributions, True

        else:
            return sol_set, obj_func, contributions, False  # Return False if
            # all arguments remains the same

    def complete_a_local_search_by_exchange(self, instance, sol_set,
                                            contributions, obj_func):
        """
        Performs a complete local search.
        """
        index = 0

        while index < len(sol_set):

            sol_set, obj_func, contributions, flag = self.exchange_component_search(
                instance,  sol_set, index,  contributions, obj_func)

            # print(index)
            if flag:
                index = 0
            else:
                index += 1

        return sol_set, obj_func

    def iterate_GRASP(self, instance, alpha, size=25):
        """
        Performs a complete GRASP iteration on the instance `instance` with
        random parameter `alpha`
        """

        sol_set, obj_func, contributions = self.construct_solution(instance,
                                                                   alpha, size)

        sol_set, obj_func = self.complete_a_local_search_by_exchange(
            instance, sol_set, contributions, obj_func)

        return sol_set, obj_func

    def perform_during(self, instance, alpha, time_max, size=25):
        """
        Keeps GRASP algorithm performing during time_max seconds.
        """
        best_sol = 0
        t_initial = time.time()
        t_performing = 0
        # ite_num = 0
        bests_list = [0]
        times_list = [0]
        while t_performing < time_max:
            iteration = self.iterate_GRASP(instance, alpha, size)[1]

            if iteration > best_sol:
                print(best_sol, "at", time.time()-t_initial,
                      "seconds with alpha ", alpha)
                best_sol = iteration
                bests_list.append(best_sol)
                times_list.append(time.time()-t_initial)
            else:
                pass
            t_performing = time.time() - t_initial
            # ite_num += 1
            # print(f'iteration: {ite_num}')

        return bests_list, times_list
