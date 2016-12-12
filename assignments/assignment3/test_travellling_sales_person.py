import unittest
import travelling_sales_person

class TestTSP(unittest.TestCase):

    def setUp(self):
        self.tsp = travelling_sales_person.TravellingSalesPerson('test.tsp.txt')

    def tearDown(self):
        pass

    def test_calc_distance(self):
        self.assertAlmostEqual(self.tsp.calc_distance('1', '2'), 5) 
        
    def test_get_distance(self):
        self.assertAlmostEqual(self.tsp.get_distance('1', '2'), 5)
        self.assertAlmostEqual(self.tsp.distance_dict['1']['2'], 5)
    
    def test_get_distance_route(self):
        route = ['1', '5', '2']
        dis1 = self.tsp.get_distance_route(route)
        dis2 = (self.tsp.get_distance('1', '5') + 
                self.tsp.get_distance('5', '2') +
                self.tsp.get_distance('2', '1'))
        self.assertAlmostEqual(dis1, dis2)

if __name__ == '__main__':
    unittest.main()
