import numpy as np 
import travelling_sales_person

def cooling_linear_gen(initial_T, rate):
    """generate decreasing temp for sim annealing
    
    params:
        rate: number between 0 and 1
    """
    if rate > 1 or rate < 0:
        raise ValueError("uexpected rate for cooling schedule")
    T = initial_T
    while True:
        yield T
        T = rate*T

def cooling_log_gen(C):
    count = 1
    while True:
        temp = C * np.log(count+1)
        yield 1/temp
        count += 1

class SimulatedAnnealingTSP():
    def __init__(self, tsp, cooling_gen):
       self.tsp = tsp 
       self.route = np.array(list(tsp.city_dict.keys()))
       self.route = np.random.permutation(self.route)
       self.cooling_gen = cooling_gen

    def initialise(self):
        self.route = np.array(list(tsp.city_dict.keys()))
        self.route = np.random.permutation(self.route)

    def simulate(self, steps):
        distance = self.tsp.get_distance_route(self.route)
        for i in range(steps):
            new_route = self.permute_lin_2_op(self.route) 
            distance_new = self.tsp.get_distance_route(new_route)
            if distance_new <= distance:
                distance = distance_new
                self.route = new_route
            else:
                u = np.random.uniform()
                change_chance = np.exp((distance-distance_new) / next(self.cooling_gen))
                if change_chance > u:
                    distance = distance_new
                    self.route = new_route

    @staticmethod 
    def permute_lin_2_op(route):
        num1 = np.random.choice(np.arange(1, route.shape[0], 1))
        num2 = np.random.choice(np.arange(1, route.shape[0], 1))
        while np.absolute(num1-num2) < 1:
            num2 = np.random.choice(np.arange(0, route.shape[0], 1))
        
        if num1>num2:
            num1, num2 = num2, num1
        new_route = np.copy(route)
        new_route[num1:num2] = new_route[num1:num2][::-1]
        return new_route

    def get_initial_temp(self, num_samples=100):
        route = np.array(list(tsp.city_dict.keys()))
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            route = np.random.permutation(self.route)
            samples[i]=self.tsp.get_distance_route(route)

        return 2*np.std(samples) / -np.log(0.8)

if __name__=='__main__':
    tsp = travelling_sales_person.TravellingSalesPerson('TSP-Configurations/eil51.tsp.txt')
    params = {
        'initial_temp':1200,
        'rate': 0.99995
    }
        
    cooling_gen_lin = cooling_linear_gen(params['initial_temp'], params['rate'])
    cooling_gen_log = cooling_log_gen(1/params['initial_temp'])
    sa = SimulatedAnnealingTSP(tsp, cooling_gen_lin)
    sa.simulate(300000)
    print(sa.get_initial_temp())
    print(sa.route)
    print(tsp.get_distance_route(sa.route))
    
    

