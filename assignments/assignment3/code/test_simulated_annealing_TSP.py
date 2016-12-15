import unittest
from itertools import permutations
import numpy as np
import travelling_sales_person
import simulated_annealing_TSP as sim

class TestSimulatedAnnealing(unittest.TestCase):

    def setUp(self):
        self.tsp = travelling_sales_person.TravellingSalesPerson('test.tsp.txt')

    def test_simulate(self):
        params = {
            'initial_temp': self.tsp.TSP_dict["DIMENSION"]*4 / -np.log(0.8),
            'rate': 0.95
        }
        
        cooling_gen = sim.cooling_linear_gen(params['initial_temp'], params['rate'])
        sa = sim.SimulatedAnnealingTSP(self.tsp, cooling_gen)
        sa.simulate(1000)
        min_route = None
        min_distance = 10**10
        for perm in permutations(list(self.tsp.city_dict.keys())):
            new_distance = self.tsp.get_distance_route(list(perm))
            if min_distance > new_distance:
               min_distance = new_distance
               min_route = list(perm)

        self.assertAlmostEqual(min_distance, self.tsp.get_distance_route(sa.route))
        sa = sim.SimulatedAnnealingTSP(self.tsp, cooling_gen)
        sa.simulate(1000)
        self.assertAlmostEqual(min_distance, self.tsp.get_distance_route(sa.route))
        sa = sim.SimulatedAnnealingTSP(self.tsp, cooling_gen)
        sa.simulate(1000)
        self.assertAlmostEqual(min_distance, self.tsp.get_distance_route(sa.route))

    def test_cooling_log_gen(self):
        cooling_log_gen = sim.cooling_log_gen(1/20)
    
 
    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
