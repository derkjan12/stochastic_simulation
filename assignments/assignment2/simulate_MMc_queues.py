import simpy
import numpy as np

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

wait_times = []

class SimulateQueue():
    def __init__(self, burn_in_time):
        self.wait_times = []            
        self.burn_in_time = burn_in_time

    def customer(self, env, count, counters, service_time, create_time, service_mode):
        if service_mode=='shortest_first':
            priority = service_time
        else:    
            priority = count

        yield env.timeout(create_time)    
        start = env.now
        with counters.request(priority=priority) as req:
            yield req
            end = env.now
            if env.now > self.burn_in_time:
                self.wait_times.append(end-start)
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

def simulate_RBM(batches, replications, burn_in_time, args_to_set_waiting_times):
    """ Use RBM sampling and return estimate of the mean and standard variance"""
    mean_waiting_times_li = []    
    batch_size = int(args_to_set_waiting_times[0]/batches)
    for i in range(replications):
        sim_queue = SimulateQueue(burn_in_time)
        sim_queue.set_waiting_times(*args_to_set_waiting_times)
        wait_times_list = [] 
        for i in range(batches-1):
            wait_times_list.append(
                sim_queue.waiting_times[batch_size*i:batch_size*(i+1)]
            )           
        wait_times_list.append(
            sim_queue.waiting_times[batch_size*(batches-1):]
        ) 
        mean_waiting_times_li.append(np.mean(wait_times_list, 0))

    return np.mean(mean_waiting_times_li), np.std(mean_waiting_times_li) 

if __name__=='__main__':
    """
    notes:service_time is assumed to have rate 1 and then load and capacity are 
        used to calculate the arrival rate
    """

    sim_queue = SimulateQueue(0)
    sim_queue.set_waiting_times(number_of_customers=10, capacity=2, 
        arrival_time_gen=get_arrival_times(get_exponential, 1.95),
        service_time_gen=get_service_times(get_exponential, 1), 
        service_mode ='FIFO'
    )

    print(sim_queue.wait_times)


