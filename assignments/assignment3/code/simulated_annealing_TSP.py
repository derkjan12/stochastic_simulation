import numpy as np 
#import matplotlib.pyplot as plt
#import seaborn as sns
import json
import time
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
        cities = np.arange(0, route.shape[0], 1)
        num1 = np.random.choice(cities)
        num2 = np.random.choice(cities)
        while np.absolute(num1-num2)<1 or np.absolute(num1-num2)==route.shape[0]: 
            num1 = np.random.choice(cities)
            num2 = np.random.choice(cities)
        
        if num1>num2:
            num1, num2 = num2, num1
        new_route = np.copy(route)
        new_route[num1:num2+1] = new_route[num1:num2+1][::-1]
        return new_route

    def get_initial_temp(self, num_samples=100):
        route = np.array(list(tsp.city_dict.keys()))
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            route = np.random.permutation(self.route)
            samples[i]=self.tsp.get_distance_route(route)

        return 2*np.std(samples) / -np.log(0.8)

    def create_plot_data_distances(self, steps):
        distances = []
        distance = self.tsp.get_distance_route(self.route)
        for i in range(steps):
            new_route = self.permute_lin_2_op(self.route) 
            distance_new = self.tsp.get_distance_route(new_route)
            if distance_new <= distance:
                distance = distance_new
                self.route = new_route
                no_change_count = 0
            else:
                u = np.random.uniform()
                change_chance = np.exp((distance-distance_new) / next(self.cooling_gen))
                if change_chance > u:
                    distance = distance_new
                    self.route = new_route
                    no_change_count = 0
                else:
                    no_change_count += 1
            
            if no_change_count > 2000:
                break
            distances.append(distance)

        return distances

    def plot_distances(self, steps):
        data = self.create_plot_data_distances(steps)
        print(tsp.get_distance_route(self.route))
        plt.plot(data)
        plt.show()

class InitialTemp():

    def __init__(self, sample_size, tsp):
        self.sample_size = sample_size
        self.tsp = tsp
        self.samples = self.set_samples(sample_size)

    def get_initial_temp_own_attempt(self, num_samples=100):
        route = np.array(list(tsp.city_dict.keys()))
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            route = np.random.permutation(self.route)
            samples[i]=self.tsp.get_distance_route(route)

        return 2*np.std(samples) / -np.log(0.8)

    def get_temp_max_diff(self):
        return max(self.samples)-min(self.samples)

    def get_temp_std(self, K):
        return np.std(self.samples) * K

    def get_temp_acceptance_ratio(self, ratio, epsilon=10**-2, p=1):
        sample_pair_li = []
        for i in range(int(self.sample_size/2)):
            route = np.array(list(tsp.city_dict.keys()))
            route = np.random.permutation(route)
            new_route = SimulatedAnnealingTSP.permute_lin_2_op(route)
            distance_old = tsp.get_distance_route(route)
            distance_new =  tsp.get_distance_route(new_route)
            if distance_old < distance_new:
                distance_old, distance_new = distance_new, distance_old
                
            sample_pair_li.append((distance_old, distance_new))
            
        min_distance = min(list(zip(*sample_pair_li))[1])
        T = 100
        for i in range(500):
            if np.isnan(T):
                raise ValueError
                
            sum_min = 0
            sum_max = 0
            

            for max_E, min_E in sample_pair_li:
                sum_min += np.exp((-min_E+min_distance)/T)
                sum_max += np.exp((-max_E+min_distance)/T)

            
            #print("min {}, max {}".format(sum_min, sum_max))
            chi_T = sum_max/sum_min 
            if np.absolute(chi_T - ratio) < epsilon:
                return T
            else:
                T = T * (np.log(chi_T)/np.log(ratio))**p

        print("the value did not converge")
        return T

    def set_samples(self, sample_size):
        self.samples = []
        route = np.array(list(tsp.city_dict.keys()))
        route = np.random.permutation(route)
        for i in range(sample_size):
            self.samples.append(self.tsp.get_distance_route(route))
            route = SimulatedAnnealingTSP.permute_lin_2_op(route)

        return self.samples

def simulate(tsp, initial_temp, rate, steps, samples):
    params_small = {
        'initial_temp':initial_temp,
        'rate': rate
    }

    final_distance_li = []
    for i in range(samples):
        cooling_gen_lin = cooling_linear_gen(params_small['initial_temp'], 
                                             params_small['rate'])
        sa = SimulatedAnnealingTSP(tsp, cooling_gen_lin)
        start = time.time()
        sa.simulate(steps)
        print("simulation took {} seconds".format(time.time()-start))
        distance = tsp.get_distance_route(sa.route)
        print("final distance {}".format(distance))
        final_distance_li.append(distance)

    return final_distance_li

def batch_mean_sampling_init_temp(tsp):
    batches = 30
    samples = 30
    max_diff_mean_li = []
    std_estimation_mean_li = []
    acceptance_ratio_mean_li = []
    for i in range(batches):
        max_diff_li = []
        std_estimation_li = []
        acceptance_ratio_li = []
        for j in range(samples):
            init_temp = InitialTemp(500, tsp)
            max_diff_li.append(init_temp.get_temp_max_diff())
            std_estimation_li.append(init_temp.get_temp_std(7.5))
            acceptance_ratio_li.append(init_temp.get_temp_acceptance_ratio(0.9))
        max_diff_mean_li.append(np.mean(max_diff_li))
        std_estimation_mean_li.append(np.mean(std_estimation_li))
        acceptance_ratio_mean_li.append(np.mean(acceptance_ratio_li))

    print_mean_std(max_diff_mean_li, "max diff")
    print_mean_std(std_estimation_mean_li, "std estimation")
    print_mean_std(acceptance_ratio_mean_li, "acceptance ratio")

def print_mean_std(li, name):
    print("for {} the mean is {} and the std {}".format(name, np.mean(li), np.std(li))) 

if __name__=='__main__':
    #tsp = travelling_sales_person.TravellingSalesPerson('TSP-Configurations/eil51.tsp.txt')
    tsp = travelling_sales_person.TravellingSalesPerson('TSP-Configurations/a280.tsp.txt')
    
    #batch_mean_sampling_init_temp(tsp)

    """
    distance_dict = {}
    for init_temp in np.arange(500, 1500, 500):
        distance_dict[str(init_temp)] = simulate(tsp, init_temp, 0.99, int(10**3), 10)
        
    with open('initial_temp_middle.json', 'w') as f:
        json.dump(distance_dict, f)
    """

    """
    params = {
        'initial_temp':1200,
        'rate': 0.999975
    }
    cooling_gen_lin = cooling_linear_gen(params['initial_temp'], params['rate'])
    sa = SimulatedAnnealingTSP(tsp, cooling_gen_lin)
    sa.plot_distances(700000)
    """
