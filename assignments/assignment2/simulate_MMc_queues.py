import simpy
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
import collections

def get_arrival_times(inter_arrival_time_func, *args):
    last_arrival_time = 0    
    while True:    
        last_arrival_time += inter_arrival_time_func(*args)
        yield last_arrival_time

def get_service_times(service_time_func, *args):
    while True:    
        yield service_time_func(*args)

def get_exponential(scale):
    """generates number from exponential distribution with scale is the rate (lambda)"""
    return np.random.exponential(scale=1/scale)

def get_deterministic(time):
    return time

def get_hyper_exponential(time_short, time_long, chance_first):
    """generate a number according to the hyper exponential distribution 
    
    params:
        time_short: inverse rate for first exponential distribution
        time_long: inverse rate for second exponential distriubtion
        chance_first: chance that number is drawn from first distribution 
    """
    if chance_first > np.random.uniform():
        return get_exponential(time_short)
    else:
        return get_exponential(time_long)

class SimulateQueue():
    def __init__(self, burn_in_time):
        self.waiting_times = []            
        self.burn_in_time = burn_in_time

    def customer(self, env, count, counters, service_time, create_time, service_mode):
        """create a simpy event process, aka, a generator that yields simpy events
        
        params:
            service_mode: either FIFO or shortest_first
        """
        if service_mode=='shortest_first':
            priority = service_time
        else:    
            priority = count

        yield env.timeout(create_time)    
        start = env.now
        with counters.request(priority=priority) as req:
            yield req
            end = env.now
            if count >= self.burn_in_time:
                self.waiting_times.append(end-start)
            #print("customer {} being served at {} wait time was {}".format(
            #    count, env.now, end-start)
            #)
            yield env.timeout(service_time)

    def set_waiting_times(self, number_of_customers, capacity, arrival_time_gen, 
                          service_time_gen, service_mode):
        """get waiting times of customers
        
        params:
            capacity: amount of servers being  used
            service_time_gen: generator object yielding service times
            arrival_time_gen: generator object yielding arrival times
            serice_mode: either FIFO or shortest_first
        """    
        
        env = simpy.Environment()
        counters = simpy.PriorityResource(env, capacity=capacity)
        for i in range(number_of_customers):
            env.process(self.customer(
                env, i, counters, next(service_time_gen), next(arrival_time_gen), 
                service_mode
                )
            )

        env.run()

def get_chance_to_wait_MMc(load, c):
    """calculate the chance that you have to wait at all

    params:
        load: the load in the system- arrival rate/(servers*service_rate)
        c: amount of servers in the system 
    """
    first = ((c*load)**c)/math.factorial(c)
    second = (1-load) * sum([((c*load)**n)/math.factorial(n) for n in range(c)])
    #print([((c*load)**n)/math.factorial(n) for n in range(c-1)])
    #print("first {} second {}".format(first, second))
    return (first/(second+first))

def get_theoretical_average_waiting_time_MMc(load, servers, service_rate=1):
    chance_to_wait = get_chance_to_wait_MMc(load, servers)
    return chance_to_wait * (1/(1-load)) * (1/(servers*service_rate))

def simulate_RBM(batch_size, replications, burn_in_time, dict_args_to_set_waiting_times):
    """ Use RBM sampling and return estimate of the mean and standard variance"""
    num_customers = dict_args_to_set_waiting_times['number_of_customers']
    samples = num_customers - burn_in_time    
    mean_waiting_times_li = []    
    for i in range(replications):
        sim_queue = SimulateQueue(burn_in_time)
        sim_queue.set_waiting_times(**dict_args_to_set_waiting_times)
        wait_times = sim_queue.waiting_times
        if len(sim_queue.waiting_times) != samples:
            raise ValueError("incorrect amount of samples taken")         
        if samples%batch_size == 0:        
            batched_waiting_times = [wait_times[i*batch_size:(i+1)*batch_size] 
                                     for i in range(int(samples/batch_size))]       
        else:
            print("not divisble number of samples")
            whole_batches = int(samples/batch_size)        
            batched_waiting_times = [wait_times[i*batch_size:(i+1)*batch_size] 
                                     for i in range(whole_batches)]
            #for count, sample in enumerate(wait_times[int(whole_batches*batch_size):]):
            #    batched_waiting_times[count].extend([sample])

        mean_waiting_times_li.extend([np.mean(li) for li in batched_waiting_times])

    return np.mean(mean_waiting_times_li), np.std(mean_waiting_times_li) 

def batch_sample_load(file_name, load_li, batch_size, replications, 
                      burn_in_time, args_set_waiting_times):
    """estimates mean and std waiting time using RBM sampling writes output to file
        
    params:
        load_li: list with loads to sample from
    """
    
    mean_std_per_load_dict = {}
    for load in load_li:
        args_set_waiting_times['arrival_time_gen'] = get_arrival_times(
            get_exponential, load * args_set_waiting_times['capacity'] 
        )
        mean_std_per_load_dict[load] = simulate_RBM(
            batch_size, replications, burn_in_time, args_set_waiting_times
        )
    
    with open(file_name, 'w') as f:
        json.dump(mean_std_per_load_dict, f, indent=2)

def plot_mean_std(plot_range, mean_li, std_li, title, x_label, y_label, data2=None):
    """creates a plot displaying mean and standard deviation of data1
       and displaying data2 as a line 

    params:
        data2: list with values for same plotrange
    """

    lower_bound = np.array(mean_li) - 1.96*np.array(std_li)
    upper_bound = np.array(mean_li) + 1.96*np.array(std_li)

    print(plot_range)
    
    plt.plot(plot_range, mean_li, label='average_waiting_time')
    if data2 is not None:
        plt.plot(plot_range, data2, label='theoretical_estimate')
    plt.fill_between(plot_range, lower_bound, upper_bound, 
                     label='95% confidence interval', alpha=0.2)
    #plt.axis([0, 1, 0, 3.5])    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()

def get_plotting_data(file_name):
    with open(file_name, 'r') as f:
        dict_val = json.load(f)
        
    ord_dict_val = collections.OrderedDict(sorted(dict_val.items(), key= lambda x: x[0]))
    plot_range = list(ord_dict_val.keys()) 
    plot_range = np.array([float(i) for i in plot_range])
    mean_li = list(list(zip(*ord_dict_val.values()))[0])
    std_li = list(list(zip(*ord_dict_val.values()))[1])
    return plot_range, mean_li, std_li

if __name__=='__main__':
    """
    notes:service_time is assumed to have rate 1 and then load and capacity are 
        used to calculate the arrival rate
    """

    args_dict = {
        'number_of_customers':110000,
        'capacity':1,
        'arrival_time_gen':get_arrival_times(get_exponential, 1.90),
        'service_time_gen':get_service_times(get_hyper_exponential, 2, 1/3, 0.8),
        'service_mode':'FIFO'
    }

    load_li = np.arange(0.1, 1, 0.1)
    #load_li_hyper_exponential = np.arange(0.1,1,0.1)/2

    batch_sample_load("MM1_FIFO_0.1_0.9_long_tail_service_times.json",
                      load_li, batch_size=10000, replications=5, 
                      burn_in_time=10000, args_set_waiting_times=args_dict)

    """
    title = "M/M/4 queue FIFO scheduling average waiting times"
    plot_data = get_plotting_data("MM4_FIFO_ex1.json")
    theoret_est = [get_theoretical_average_waiting_time_MMc(i, 4) 
                       for i in plot_data[0]]
    plot_mean_std(*plot_data, title, "loads", "average_waiting_times", theoret_est)
    """

    """
    print(simulate_RBM(batch_size=50000, replications=5, burn_in_time=20000, 
          dict_args_to_set_waiting_times=args_dict))
        
    sim_queue = SimulateQueue(0)
    sim_queue.set_waiting_times(number_of_customers=11, capacity=2, 
        arrival_time_gen=get_arrival_times(get_exponential, 1.95),
        service_time_gen=get_service_times(get_exponential, 1), 
        service_mode ='FIFO'
    )

    #print(sim_queue.waiting_times)
    """

