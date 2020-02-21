import copy
import sys
import random
import math
import matplotlib.pyplot as plt
import time

import numpy as np
import scipy.spatial.distance as dist

class TSPTabu:

    def __init__(self, n_cities, n_deliveries, seed=7):
        self.cities, self.dm, self.deliveries = self.generate_dm(n_cities, n_deliveries, seed)
        self.cities2, self.dm, self.deliveries = self.generate_dm(n_cities, n_deliveries, seed)
        self.min_dist = sys.maxsize
        self.shortest_path = []
        self.neighbours = {}
        self.first_solution = []
        self.distance_of_first_solution = sys.maxsize
        self.neighborhood = []

    def generate_neighbours(self):
        """
        Generate a dictionary of neighbours

        """

        for i in range(0, len(self.dm)):
            for j in range(0, len(self.dm[0])):
                if i != j:
                    temp_neighs = []
                    if i not in self.neighbours:
                        temp_neighs.append([j, self.dm[i][j]])
                        self.neighbours[i] = temp_neighs
                    else:
                        self.neighbours[i].append([j, self.dm[i][j]])
                    temp_neighs = []
                    if j not in self.neighbours:
                        temp_neighs.append([i, self.dm[i][j]])
                        self.neighbours[j] = temp_neighs
                    else:
                        self.neighbours[j].append([i, self.dm[i][j]])

        # remove duplicates (which we got due to matrix structure)
        neighours_no_dup = {}
        for key in self.neighbours:
            dup_n = set(tuple(element) for element in self.neighbours[key])
            neighours_no_dup[key] = [list(t) for t in set(tuple(element) for element in dup_n)]
        self.neighbours = neighours_no_dup

    def generate_first_solution(self):
        """
        Generate a first best solution to let tabu start
        """

        start_city = 0
        end_city = start_city

        current = start_city

        temp_dist = 0

        while current not in self.first_solution:
            min = 10000
            for k in self.neighbours[current]:
                if int(k[1]) < min and k[0] not in self.first_solution:
                    min = k[1]
                    best_city = k[0]

            self.first_solution.append(current)
            temp_dist = temp_dist + int(min)
            current = best_city

        self.first_solution.append(end_city)

        position = 0
        for k in self.neighbours[self.first_solution[-2]]:
            if k[0] == start_city:
                break
            position += 1

        self.distance_of_first_solution = (temp_dist
                                           + int(self.neighbours[self.first_solution[-2]][position][1])
                                           - 10000)

    def generate_first_solution_convex(self):

        self.cities2.sort(key=lambda x: (x[0], -x[1]))
        pts = self.cities2

        for point in pts:
            index = self.cities.index(point)
            self.first_solution.append(index)
        self.first_solution.append(self.cities.index(pts[0]))

        print(self.first_solution)
        self.shortest_path = self.first_solution
        self.show()

    def find_neighborhood(self, solution):
        """
        Find a neighbordhood of the solution with a one to one exchange method
        """

        for n in solution[1:-1]:
            first_index = solution.index(n)
            for j in solution[1:-1]:
                second_index = solution.index(j)
                if n == j:
                    continue

                tmp = copy.deepcopy(solution)
                tmp[first_index] = j
                tmp[second_index] = n

                dist = 0

                for k in tmp[:-1]:
                    next_city = tmp[tmp.index(k) + 1]
                    for i in self.neighbours[k]:
                        if i[0] == next_city:
                            dist = dist + int(i[1])
                tmp.append(dist)

                if tmp not in self.neighborhood:
                    self.neighborhood.append(tmp)

        last_index = len(self.neighborhood[0]) - 1
        self.neighborhood.sort(key=lambda x: x[last_index])

    def tabu_search(self, iterations, size):
        """
        TSP with badu
        """
        current_iter = 1
        solution = self.first_solution
        tabu_list = []
        best_cost = self.distance_of_first_solution
        best_solution_ever = solution

        while current_iter <= iterations:
            self.find_neighborhood(solution)
            index_of_best_solution = 0
            best_solution = self.neighborhood[index_of_best_solution]
            best_cost_index = len(best_solution) - 1

            found = False
            while found is False:
                i = 0
                # exchange every node
                while i < len(best_solution) - 1:
                    if best_solution[i] != solution[i]:
                        first_exchange_city = best_solution[i]
                        second_exchange_city = solution[i]
                        break
                    i = i + 1

                # test if option is not tabu
                if [first_exchange_city, second_exchange_city] not in tabu_list and \
                        [second_exchange_city, first_exchange_city] not in tabu_list:
                    tabu_list.append([first_exchange_city, second_exchange_city])
                    found = True
                    solution = best_solution[:-1]
                    cost = self.neighborhood[index_of_best_solution][best_cost_index]
                    if cost < best_cost:
                        best_cost = cost
                        best_solution_ever = solution
                else:
                    if index_of_best_solution == len(self.neighborhood) - 1:
                        break
                    index_of_best_solution = index_of_best_solution + 1
                    best_solution = self.neighborhood[index_of_best_solution]

            # if tabu list is too big, pop list
            if len(tabu_list) >= size:
                tabu_list.pop(0)

            current_iter = current_iter + 1

        self.min_dist = best_cost
        self.shortest_path = best_solution_ever

    def get_shortest_path_and_min_dist(self):
        return self.shortest_path, self.min_dist

    def generate_dm(self, n_cities, n_deliveries, seed):
        cities = []
        if seed is not None:
            random.seed(seed)
        for _ in range(n_cities):
            cities.append([random.randint(0, 100), random.randint(0, 100)])

        dm = []
        for i in cities:
            dm_row = []
            for j in cities:
                dm_row.append(int(math.sqrt(((i[0] - j[0]) ** 2) + ((i[1] - j[1]) ** 2))))
            dm.append(dm_row)

        deliveries = []

        for i in range(1, n_cities):
            if random.randint(0, 1) == 0:
                deliveries.append([i, 0])
        for _ in range(n_deliveries):
            deliveries.append([random.randint(1, n_cities - 1), random.randint(0, 1000)])

        return cities, dm, deliveries

    def generate_dm_alt(self, n_cities, n_deliveries, seed, do_print=False):
        cities = []
        if seed is not None:
            random.seed(seed)
        for i in range(n_cities):
            if i % 2 == 0:
                cities.append([0, i/2 *10])
            else:
                cities.append([10, (i-1)/2 *10])
        if do_print:
            if n_cities % 2 == 0:
                print("Optimal solution = " + str(n_cities * 10))
            else:
                print("Optimal solution = " + str((n_cities-1) * 10 + math.sqrt(2)*10))

        dm = []
        for i in cities:
            dm_row = []
            for j in cities:
                dm_row.append(int(math.sqrt(((i[0] - j[0]) ** 2) + ((i[1] - j[1]) ** 2))))
            dm.append(dm_row)

        deliveries = []

        for i in range(1, n_cities):
            if random.randint(0, 1) == 0:
                deliveries.append([i, 0])
        for _ in range(n_deliveries):
            deliveries.append([random.randint(1, n_cities - 1), random.randint(0, 1000)])

        return cities, dm, deliveries

    def show(self):
        cities_x, cities_y = map(list, zip(*self.cities))
        sorted_cities = []
        for i in self.shortest_path:
            sorted_cities.append(self.cities[i])
        sorted_cities_x, sorted_cities_y = map(list, zip(*sorted_cities))
        plt.scatter(cities_x, cities_y)
        plt.plot(sorted_cities_x, sorted_cities_y)
        plt.scatter(cities_x[0], cities_y[0], 80)
        plt.show()


class TSP:
    def __init__(self, n_cities, n_deliveries, seed=7):
        self.cities, self.dm, self.deliveries = self.generate_dm(n_cities, n_deliveries, seed)
        self.min_dist = sys.maxsize
        self.shortest_path = []

    def search_shortest_path(self, strategy=0, print_enabled=False):
        if strategy == 0:
            self.search_shortest_path_with_pruning(0, [], 0, print_enabled)
        elif strategy == 1:
            self.find_shortest_path_with_pruning_and_sorting(self.dm, 0, [], self.cities, 0, print_enabled, 1)
        else:
            print("Given strategy is undefined")

    def search_shortest_path_with_pruning(self, start_city, path, distance, print_enabled):
        path.append(start_city)

        if len(path) > 1:
            distance += self.dm[path[-2]][start_city]
            if self.min_dist is not None and distance > self.min_dist:
                return

        if len(self.cities) == len(path):
            path.append(path[0])
            distance += self.dm[path[-2]][path[0]]
            if self.min_dist is not None and distance >= self.min_dist:
                return
            if print_enabled:
                print(path, distance, self.min_dist)
            self.min_dist = distance
            self.shortest_path = path
            return

        for city in range(len(self.cities)):
            if city not in path:
                self.search_shortest_path_with_pruning(city, list(path), int(distance), print_enabled)

    def find_shortest_path_with_pruning_and_sorting(self, dm, chosen_city, path, cities, distance, print_enabled, tries):
        scm = []
        for row in dm:
            scm.append(sorted(range(len(row)), key=lambda k: row[k]))
        self.find_shortest_path_with_pruning_and_sorting_helper(dm, scm, chosen_city, path, cities, distance, print_enabled,
                                                           tries)

    def find_shortest_path_with_pruning_and_sorting_helper(self, dm, scm, chosen_city, path, cities, distance, print_enabled,
                                                           tries):
        # Add way point
        path.append(chosen_city)
        # Calculate path length from current to last node
        if len(path) > 1:
            distance += dm[path[-2]][chosen_city]
            if self.min_dist is not None and distance > self.min_dist:
                return

        # If path contains all cities and is not a dead end,
        # add path from last to first city and return.
        if (len(cities) == len(path)):
            path.append(path[0])
            distance += dm[path[-2]][path[0]]
            if self.min_dist is not None and distance >= self.min_dist:
                return
            self.min_dist = distance
            self.shortest_path = path
            if print_enabled:
                print(path, distance, self.min_dist)
                self.show()
            return

        # Fork paths for all possible cities not yet used
        curr_tries = tries
        for city in scm[chosen_city]:
            if city not in path and curr_tries != 0:
                self.find_shortest_path_with_pruning_and_sorting_helper(dm, scm, city, list(path), cities, int(distance),
                                                                   print_enabled, tries)
                curr_tries -= 1

    def get_shortest_path_and_min_dist(self):
        return self.shortest_path, self.min_dist

    def generate_dm(self, n_cities, n_deliveries, seed):
        cities = []
        if seed is not None:
            random.seed(seed)
        for _ in range(n_cities):
            cities.append([random.randint(0, 100), random.randint(0, 100)])

        dm = []
        for i in cities:
            dm_row = []
            for j in cities:
                dm_row.append(int(math.sqrt(((i[0] - j[0]) ** 2) + ((i[1] - j[1]) ** 2))))
            dm.append(dm_row)

        deliveries = []

        for i in range(1, n_cities):
            if random.randint(0, 1) == 0:
                deliveries.append([i, 0])
        for _ in range(n_deliveries):
            deliveries.append([random.randint(1, n_cities - 1), random.randint(0, 1000)])

        return cities, dm, deliveries

    def show(self):
        cities_x, cities_y = map(list, zip(*self.cities))
        sorted_cities = []
        for i in self.shortest_path:
            sorted_cities.append(self.cities[i])
        sorted_cities_x, sorted_cities_y = map(list, zip(*sorted_cities))
        plt.scatter(cities_x, cities_y)
        plt.plot(sorted_cities_x, sorted_cities_y)
        plt.scatter(cities_x[0], cities_y[0], 80)
        plt.show()

def find_shortest_path_with_pruning(distance_matrix, chosen_city, path, cities, distance, print_enabled):
    global min_dist, shortest_path
    # Add way point
    path.append(chosen_city)
    # Calculate path length from current to last node
    if len(path) > 1:
        distance += distance_matrix[path[-2]][chosen_city]
        if min_dist is not None and distance > min_dist:
            return

    # If path contains all cities and is not a dead end,
    # add path from last to first city and return.
    if (len(cities) == len(path)):
        path.append(path[0])
        distance += distance_matrix[path[-2]][path[0]]
        if min_dist is not None and distance >= min_dist:
            return
        if print_enabled:
            print(path, distance, min_dist)
        min_dist = distance
        shortest_path = path
        return

    # Fork paths for all possible cities not yet used
    for city in cities:
        if city not in path:
            find_shortest_path_with_pruning(distance_matrix, city, list(path), cities, int(distance), print_enabled)

def find_shortest_path_with_pruning_and_sorting_smart(dm, chosen_city, path, cities, distance, print_enabled):
    global min_dist_a, shortest_path_a, chosen_city_global_a
    min_dist_a = None
    shorterst_path_a = None
    shortest_distance_a = sys.maxsize
    for tries in range(1, len(dm[0])):
        start = time.time()
        find_shortest_path_with_pruning_and_sorting(dm, 0, [], cities, 0, False, tries)
        if min_dist_a >= shortest_distance_a:
            break
        else:
            shortest_distance_a = min_dist_a


def find_shortest_delivery_time(dm, curr_city, path, cities, time, print_enabled, dvs):
    latest_cargo_time = dvs[0][1]
    cargo_path = []
    for city, dlv_time in dvs:
        if dlv_time > latest_cargo_time:
            if curr_city != 0:
                time += dm[curr_city][0]
                path.append(0)
                curr_city = 0
                cargo_path = []
            latest_cargo_time = time
        time += dm[curr_city][city]
        path.append(city)
        curr_city = city
    show_path(cities, path)
    print(time)

def generate_dm(n_cities, n_deliveries, seed):
        cities = []
        if seed is not None:
            random.seed(seed)
        for _ in range(n_cities):
            cities.append([random.randint(0, 100), random.randint(0, 100)])

        dm = []
        for i in cities:
            dm_row = []
            for j in cities:
                dm_row.append(int(math.sqrt(((i[0] - j[0]) ** 2) + ((i[1] - j[1]) ** 2))))
            dm.append(dm_row)

        deliveries = []

        for i in range(1, n_cities):
            if random.randint(0, 1) == 0:
                deliveries.append([i, 0])
        for _ in range(n_deliveries):
            deliveries.append([random.randint(1, n_cities - 1), random.randint(0, 1000)])

        return cities, dm, deliveries

def show_path(cities, shortest_path):
    cities_x, cities_y = map(list, zip(*cities))
    sorted_cities = []
    for i in shortest_path:
        sorted_cities.append(cities[i])
    sorted_cities_x, sorted_cities_y = map(list, zip(*sorted_cities))
    plt.scatter(cities_x, cities_y)
    plt.plot(sorted_cities_x, sorted_cities_y)
    plt.scatter(cities_x[0], cities_y[0], 80)
    plt.show()


def find_shortest_delivery_time_with_tsp(dm, curr_city, path, cities, time, print_enabled, dvs):
    cargo = []
    latest_cargo_time = dvs[0][1]
    for city, dlv_time in dvs:
        if dlv_time <= latest_cargo_time:
            cargo.append(city)
        else:
            if len(cargo) == 0:
                latest_cargo_time = dlv_time
                cargo.append(city)
            # deliver all cargo using tsp
            global min_dist, shortest_path, chosen_city_global
            min_dist = None
            shorterst_path = None
            shortest_distance = sys.maxsize
            find_shortest_path_with_pruning(dm, curr_city, [], cargo, 0, False)
            time += min_dist
            path.extend(shortest_path)
            latest_cargo_time = time
    show_path(cities, path)
    print(time)


def test():
    global min_dist, shortest_path, chosen_city_global
    #cities, dm = generate_dm(12, 200, 1)
    cities, dm, dvs = generate_dm(100, 500, 1)
    min_dist = None
    shorterst_path = None
    shortest_distance = sys.maxsize
    for tries in range(1, len(dm[0])):
        start = time.time()
        find_shortest_delivery_time(dm, 0, [], cities, 0, True, dvs)
        print("time:" + str(time.time() - start) + "s tries:" + str(tries) + " min_dist:" + str(min_dist))
        if min_dist >= shortest_distance:
            break
        else:
            shortest_distance = min_dist


def test_dvs():
    global min_dist, shortest_path, chosen_city_global
    cities, dm, dvs = generate_dm(20, 8, 7)
    find_shortest_delivery_time(dm, 0, [], cities, 0, True, dvs)
    find_shortest_delivery_time_with_tsp(dm, 0, [], cities, 0, True, dvs)


def run_all():
    global min_dist, shorterst_path, chosen_city_global
    cities, dm, dvs = generate_dm(12, 8, 7)

    min_dist = None
    shorterst_path = None
    start = time.time()
    find_shortest_path_with_pruning(dm, 0, [], range(len(dm)), 0, True)
    find_shortest_path_with_pruning_time = time.time() - start
    print("find_shortest_path_with_pruning: " + str(find_shortest_path_with_pruning_time) + "s")

    min_dist = None
    shorterst_path = None
    start = time.time()
    find_shortest_path_with_pruning_and_sorting(dm, 0, [], range(len(dm)), 0, True, 3)
    find_shortest_path_with_pruning_and_sorting_time = time.time() - start
    print("find_shortest_path_with_pruning_and_sorting_smart: " + str(
        find_shortest_path_with_pruning_and_sorting_time) + "s speed up of " + str(
        find_shortest_path_with_pruning_time / find_shortest_path_with_pruning_and_sorting_time) + " times compared to previous version")

def test_run():
    start = time.time()
    tsp = TSP(20, 8, 7)
    tsp.search_shortest_path(0)
    tsp.show(1, 10)


def test_run_tabu():
    for iterations in range(1, 150):
        start = time.time()
        tsp = TSPTabu(50, 8, 7)
        tsp.generate_neighbours()
        tsp.generate_first_solution()
        #tsp.generate_first_solution_convex()
        tsp.tabu_search(iterations, 10*iterations)
        #print("time: " + str(time.time() - start))
        tsp.show()
        _, mindist = tsp.get_shortest_path_and_min_dist()
        print("(" +str(mindist) + ", " + str(tsp.distance_of_first_solution) + ")", end="")
    print(" ")

test_dvs()
