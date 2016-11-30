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
    return np.random.exponential(scale=scale)

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

def customer(env, count, counters, service_time, create_time, service_mode):
    if service_mode=='shortest_first':
        priority = service_time
    else:    
        priority = count

    yield env.timeout(create_time)    
    start = env.now
    with counters.request(priority=priority) as req:
        yield req
        end = env.now
        if env.now > burn_in_time:
            wait_times.append(end-start)
        print("customer {} being served at {} wait time was {}".format(
            count, env.now, end-start)
        )
        yield env.timeout(service_time)

def set_waiting_times(burn_in_time, number_of_customers, capacity, arrival_time_gen, 
                      service_time_gen):
    """get waiting times of customers"""    
    
    env = simpy.Environment()
    counters = simpy.PriorityResource(env, capacity=capacity)
    for i in range(number_of_customers):
        env.process(customer(
            env, i, counters, next(service_time_gen), next(arrival_time_gen), 
            'shortest_first'
            )
        )

    env.run()

if __name__=='__main__':
    burn_in_time = 0
    number_of_customers = 10
    capacity = 2
    set_waiting_times(burn_in_time, 10, 1, get_arrival_times(get_exponential, 1.6), 
                      get_service_times(get_exponential, 1))

    print(wait_times)


