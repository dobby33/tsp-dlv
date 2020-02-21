import copy
import sys
import random
import math
import matplotlib.pyplot as plt
import time
import numpy as np
import os

from scipy.spatial import ConvexHull


class TSP:
    def __init__(self, cities, dm, deliveries):
        self.cities, self.dm, self.deliveries = cities, dm, deliveries
        self.active_cities = range(len(cities))
        self.min_dist = sys.maxsize
        self.shortest_path = []

    def set_active_cities(self, cities=None, all=False):
        cities = list(set(cities))
        if all:
            self.active_cities = range(len(self.cities))
        if cities is None:
            cities = []
        self.active_cities = cities

    def search_shortest_path(self, strategy=0, print_enabled=False):
        if strategy == 0:
            self.search_shortest_path_with_pruning(self.active_cities[0], [], 0, print_enabled)
        elif strategy == 1:
            self.find_shortest_path_with_pruning_and_sorting(self.dm, self.active_cities, [], self.active_cities, 0,
                                                             print_enabled, 1)
        elif strategy == 2:
            self.find_shortest_path_with_local_search()
        else:
            print("Given strategy is undefined")

    def find_shortest_path_with_local_search(self):
        self.generate_first_solution_random()
        repeat = self.local_search_2opt()
        while repeat:
            # self.show()
            repeat = self.local_search_2opt()

    def rearange_solution(self):
        index = self.shortest_path.index(0)
        tmp_path = self.shortest_path[index:]
        tmp_path.extend(self.shortest_path[:index])
        self.shortest_path = tmp_path

    def generate_first_solution(self):
        self.shortest_path = self.active_cities
        self.min_dist = self.dm[self.shortest_path[-1]][self.shortest_path[0]]
        for i in range(len(self.active_cities) - 1):
            self.min_dist += self.dm[self.shortest_path[i]][self.shortest_path[i + 1]]

    def generate_first_solution_random(self):
        self.shortest_path = list(np.random.permutation(self.active_cities))
        self.min_dist = self.dm[self.shortest_path[-1]][self.shortest_path[0]]
        for i in range(len(self.active_cities) - 1):
            self.min_dist += self.dm[self.shortest_path[i]][self.shortest_path[i + 1]]

    def local_search_2opt(self):
        for start in range(len(self.active_cities)):
            for leng in range(2, int(len(self.active_cities))):
                i = start
                j = (start + leng) % len(self.active_cities)
                diff = self.dm[self.shortest_path[(i - 1) % len(self.active_cities)]][self.shortest_path[i]] + \
                       self.dm[self.shortest_path[(j - 1) % len(self.active_cities)]][self.shortest_path[j]] - (
                               self.dm[self.shortest_path[(i - 1) % len(self.active_cities)]][
                                   self.shortest_path[(j - 1) % len(self.active_cities)]] +
                               self.dm[self.shortest_path[j]][self.shortest_path[i]])
                if diff > 0:
                    new_solution = []
                    tmp = []
                    for k in range(len(self.active_cities)):
                        if k == i:
                            new_solution.extend(tmp)
                            tmp = []
                        elif k == j:
                            new_solution.extend(tmp[::-1])
                            tmp = []
                        tmp.append(self.shortest_path[k])
                    if i < j:
                        new_solution.extend(tmp[::])
                    else:
                        new_solution.extend(tmp[::-1])
                    self.shortest_path = new_solution
                    self.min_dist -= diff
                    return True
        return False

    def evaluate(self, path):
        cost = 0
        if len(path) == len(self.cities):
            path.append(path[0])
        for node in path:
            pass

    def search_shortest_path_with_pruning(self, start_city, path, distance, print_enabled):
        path.append(start_city)

        if len(path) > 1:
            distance += self.dm[path[-2]][start_city]
            if self.min_dist is not None and distance > self.min_dist:
                return

        if len(self.active_cities) == len(path):
            path.append(path[0])
            distance += self.dm[path[-2]][path[0]]
            if self.min_dist is not None and distance >= self.min_dist:
                return
            if print_enabled:
                print(path, distance, self.min_dist)
            self.min_dist = distance
            self.shortest_path = path
            return

        dm_row = sorted(range(len(self.dm[path[-1]])), key=lambda ind: self.dm[path[-1]][ind])
        pool = sorted(set(set(dm_row) - set(path)).intersection(self.active_cities), key=dm_row.index)

        for city in pool[:min(int(1.02 ** (len(self.active_cities) - len(pool))), len(pool))]:
            if city not in path:
                self.search_shortest_path_with_pruning(city, list(path), int(distance), print_enabled)

    def find_shortest_path_with_pruning_and_sorting(self, dm, chosen_city, path, active_cities, distance, print_enabled,
                                                    tries):
        scm = []
        for row in dm:
            scm.append(sorted(active_cities, key=lambda k: row[k]))
        self.find_shortest_path_with_pruning_and_sorting_helper(dm, scm, chosen_city, path, active_cities, distance,
                                                                print_enabled,
                                                                tries)

    def find_shortest_path_with_pruning_and_sorting_helper(self, dm, scm, chosen_city, path, active_cities, distance,
                                                           print_enabled,
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
        if (len(active_cities) == len(path)):
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
                self.find_shortest_path_with_pruning_and_sorting_helper(dm, scm, city, list(path), active_cities,
                                                                        int(distance),
                                                                        print_enabled, tries)
                curr_tries -= 1

    def get_shortest_path_and_min_dist(self, make_loop=False):
        self.rearange_solution()
        if make_loop:
            if self.shortest_path[-1] != 0:
                self.shortest_path.append(0)
        return self.shortest_path, self.min_dist

    def show(self):
        cities = []
        for city in self.cities:
            cities.append([city[1], city[2]])
        cities_x, cities_y = map(list, zip(*cities))
        sorted_cities = []
        for i in self.shortest_path:
            sorted_cities.append(cities[i])
        if len(self.shortest_path) == len(self.active_cities):
            sorted_cities.append(cities[self.shortest_path[0]])
        sorted_cities_x, sorted_cities_y = map(list, zip(*sorted_cities))
        plt.scatter(cities_x, cities_y)
        plt.plot(sorted_cities_x, sorted_cities_y)
        plt.scatter(cities_x[0], cities_y[0], 80)
        plt.show()


class DLV:
    def __init__(self, cities, dm, deliveries):
        self.cities, self.dm, self.deliveries = cities, dm, deliveries
        self.deliveries = sorted(self.deliveries, key=lambda x: x[1])
        self.time = 0
        self.shortest_path = []

    def reset(self):
        self.time = 0
        self.shortest_path = []

    def greedy_search(self, enable_print=False):
        cargo = [0]
        latest_cargo_time = 0
        self.time = self.deliveries[0][1]
        for city, dlv_time in self.deliveries:
            if dlv_time <= latest_cargo_time:
                cargo.append(city)
            else:
                cargo = list(set(cargo))
                tsp = TSP(self.cities, self.dm, self.deliveries)
                tsp.set_active_cities(cities=cargo)
                tsp.search_shortest_path(2)
                shortest_path, min_dist = tsp.get_shortest_path_and_min_dist(make_loop=True)
                if enable_print:
                    tsp.show()
                self.time += min_dist
                self.shortest_path.append(shortest_path)
                latest_cargo_time = self.time
                cargo = [0, city]
                if dlv_time > self.time:
                    self.time += dlv_time - self.time
        if len(cargo) > 1:
            cargo = list(set(cargo))
            tsp = TSP(self.cities, self.dm, self.deliveries)
            tsp.set_active_cities(cities=cargo)
            tsp.search_shortest_path(2)
            shortest_path, min_dist = tsp.get_shortest_path_and_min_dist(make_loop=True)
            if enable_print:
                tsp.show()
            self.time += min_dist
            self.shortest_path.append(shortest_path)

    def reverse_greedy_search(self, enable_print=False):
        min_i, min_end = 1, sys.maxsize
        for i in range(1, len(self.cities) - 1):
            cargo = self.deliveries[i * -1:]
            get_cities = lambda x: x[0]
            cargo_cities = [0]
            cargo_cities.extend(list(map(get_cities, cargo)))
            tsp = TSP(self.cities, self.dm, self.deliveries)
            tsp.set_active_cities(cities=cargo_cities)
            tsp.search_shortest_path(2)
            shortest_path, min_dist = tsp.get_shortest_path_and_min_dist(make_loop=True)
            end = min_dist + self.deliveries[-1][1]

            cargo = self.deliveries[:i * -1]
            cargo_cities = [0]
            cargo_cities.extend(list(map(get_cities, cargo)))
            tsp = TSP(self.cities, self.dm, self.deliveries)
            tsp.set_active_cities(cities=cargo_cities)
            tsp.search_shortest_path(2)
            shortest_path_b, sub_path_dist = tsp.get_shortest_path_and_min_dist(make_loop=True)
            # Check if ready before start of next round
            end = self.deliveries[i][1] + sub_path_dist + max(0, self.deliveries[-1][1] - (
                    self.deliveries[i][1] + sub_path_dist))
            if end < min_end:
                min_i, min_end = i, end
                self.shortest_path = [shortest_path_b]
                self.shortest_path.append(shortest_path)
                self.time = min_end
            print(str(i) + " - " + str(end))

    def optimize_solution(self):
        pass

    def show(self):
        cities = []
        for city in self.cities:
            cities.append([city[1], city[2]])
        cities_x, cities_y = map(list, zip(*cities))
        plt.scatter(cities_x, cities_y)
        for loop in self.shortest_path:
            sorted_cities = [cities[0]]
            for city in loop:
                sorted_cities.append(cities[city])
            sorted_cities.append(cities[0])
            sorted_cities_x, sorted_cities_y = map(list, zip(*sorted_cities))
            plt.plot(sorted_cities_x, sorted_cities_y)
        plt.scatter(cities_x[0], cities_y[0], 80)
        plt.title("Total time = " + str(self.time))
        plt.show()


class ProblemGenerator:
    @staticmethod
    def generate_dm(n_cities, n_deliveries, max_time=1000, seed=1, start_cargo = True):
        cities = []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        city_center = 20
        i = 0
        for _ in range(0, int(n_cities * 0.8)):
            cities.append([i, random.randint(50 - city_center, 50 + city_center),
                           random.randint(50 - city_center, 50 + city_center)])
            i += 1

        while len(cities) < n_cities:
            cities.append([i, random.randint(0, 100), random.randint(0, 100)])
            i += 1

        dm = []
        for i in cities:
            dm_row = []
            for j in cities:
                dm_row.append(int(math.sqrt(((i[1] - j[1]) ** 2) + ((i[2] - j[2]) ** 2))))
            dm.append(dm_row)
        deliveries = []

        if start_cargo:
            for i in range(1, min(n_cities, n_deliveries-1)):
                if random.random() < 0.1:
                    deliveries.append([cities[i][0], 0])
        while len(deliveries) < n_deliveries-1:
            deliveries.append([cities[random.randint(1, n_cities - 1)][0], random.randint(0, max_time)])
        deliveries.append([cities[random.randint(1, n_cities - 1)][0], max_time])

        return cities, dm, deliveries


# t1 = time.time()
# cities, dm, deliveries = ProblemGenerator.generate_dm(50, 100, 500, 1, start_cargo=False)
# dlv = DLV(cities, dm, deliveries)
# # t2 = time.time()
# # dlv.greedy_search(enable_print=False)
# # t3 = time.time()
# # dlv.show()
# for i in range(10):
#     dlv.reset()
#     dlv.greedy_search(enable_print=False)
#     dlv.show()
#     print(str(dlv.time))



# print("Generation time = " + str(t2 - t1))
# print("Solve time = " + str(t3 - t2))
# print("Loops = " + str(len(dlv.shortest_path)))



for c in range(10,100, 10):
    for d in range(10, 500, 10):
        t = d*5
        sum = 0
        cities, dm, deliveries = ProblemGenerator.generate_dm(c, d, t, 1, start_cargo=True)
        dlv = DLV(cities, dm, deliveries)
        for i in range(10):
            dlv.reset()
            dlv.greedy_search(enable_print=False)
            sum += dlv.time
        print(str(sum/10))
fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(131, title='imshow: square bins')
plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

