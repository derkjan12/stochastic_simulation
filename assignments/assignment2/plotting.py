import simpy
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
import collections

def get_plotting_data(file_name):
    with open(file_name, 'r') as f:
        dict_val = json.load(f)
        
    ord_dict_val = collections.OrderedDict(sorted(dict_val.items(), key= lambda x: x[0]))
    plot_range = list(ord_dict_val.keys()) 
    plot_range = np.array([float(i) for i in plot_range])
    mean_li = np.array(list(list(zip(*ord_dict_val.values()))[0]))
    std_li = np.array(list(list(zip(*ord_dict_val.values()))[1]))
    return plot_range, mean_li, std_li

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

def plot_conf_intervals(plot_range, mean_arr1, std_arr1, mean_arr2, std_arr2,
                          title, x_label, y_label, mean_arr3=None, std_arr3=None):
    """creates a plot displaying mean and standard deviation of data1
       and displaying data2 as a line 

    params:
        data2: list with values for same plotrange
    """

    lower_bound1 = mean_arr1 - 1.96*std_arr1
    upper_bound1 = mean_arr1 + 1.96*std_arr1
    lower_bound2 = mean_arr2 - 1.96*std_arr2
    upper_bound2 = mean_arr2 + 1.96*std_arr2

    plt.plot(plot_range, mean_arr1, label='FIFO')
    plt.plot(plot_range, mean_arr2, label='shortest first')
    
    plt.fill_between(plot_range, lower_bound1, upper_bound1, 
                     label='95% conf', alpha=0.3)
    plt.fill_between(plot_range, lower_bound2, upper_bound2, 
                     label='95% conf', alpha=0.3)


    plt.plot(plot_range, mean_arr2, label='shortest first')
    plt.fill_between(plot_range, lower_bound1, upper_bound1, 
                     label='95% conf', alpha=0.3)

    plt.axis([0.1, 0.9, 0, 11])    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()

"""
title = "comparing FIFO and shortest scheduling"
x_label = "loads"
y_label = "average waiting times"
plot_data1 = get_plotting_data("MM1_FIFO_0.1_0.9.json")
_, mean_arr2, std_arr2 = get_plotting_data("MM1_shortest_first_0.1_0.9.json")

plot_conf_intervals(*plot_data1, mean_arr2, std_arr2,
                          title, x_label, y_label)

theoret_est = [get_theoretical_average_waiting_time_MMc(i, 4) 
               for i in plot_data1[0]]
"""


def plot_conf_intervals2(plot_range, title, x_label, y_label, mean_arr1, std_arr1,
                         label1, mean_arr2, std_arr2, label2, mean_arr3=None, 
                         std_arr3=None, label3=None):
    """creates a plot displaying mean and standard deviation of data1
       and displaying data2 as a line 

    params:
        data2: list with values for same plotrange
    """

    lower_bound1 = mean_arr1 - 1.96*std_arr1
    upper_bound1 = mean_arr1 + 1.96*std_arr1
    lower_bound2 = mean_arr2 - 1.96*std_arr2
    upper_bound2 = mean_arr2 + 1.96*std_arr2

    plt.plot(plot_range, mean_arr1, label=label1)
    plt.plot(plot_range, mean_arr2, label=label2)
    plt.fill_between(plot_range, lower_bound1, upper_bound1, 
                     label=label1, facecolor='green', alpha=0.3)
    plt.fill_between(plot_range, lower_bound2, upper_bound2, 
                     alpha=0.3, facecolor='red', label=label2)

    if mean_arr3 is not None and std_arr3 is not None and label3 is not None:
        lower_bound3 = mean_arr3 - 1.96*std_arr3
        upper_bound3 = mean_arr3 + 1.96*std_arr3
        plt.plot(plot_range, mean_arr3, label=label3)
        plt.fill_between(plot_range, lower_bound3, upper_bound3, 
                         alpha=0.3, facecolor='blue', label=label3)
        
    plt.axis([0.9, 0.99, -40, 200])    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.show()

"""
title = "comparing deterministic, exponential and hyper-exponential distributed service times"
x_label = "loads"
y_label = "average waiting times"

plot_data1 = get_plotting_data("MM1_FIFO_0.1_0.9.json")
_, mean_arr2, std_arr2 = get_plotting_data(
    "MM1_FIFO_0.1_0.9_long_tail_service_times.json"
)
_, mean_arr3, std_arr3 = get_plotting_data(
    "MM1_FIFO_0.1_0.9_deterministic_service_times.json"
)

plot_conf_intervals2(plot_data1[0], title, x_label, y_label, plot_data1[1], 
                    plot_data1[2], 'exponential', mean_arr2, std_arr2, 
                    'hyper-exponential', mean_arr3, std_arr3, 'deterministic')

"""

title = "comparing M/M/1, M/M/2, M/M/4 queues for load from 0.9 to 0.99"
x_label = "loads"
y_label = "average waiting times"

plot_data1 = get_plotting_data("MM1_FIFO_0.9_0.99.json")
_, mean_arr2, std_arr2 = get_plotting_data('MM2_FIFO_0.9_0.99.json')
_, mean_arr3, std_arr3 = get_plotting_data('MM4_FIFO_0.9_0.99.json')

plot_conf_intervals2(plot_data1[0], title, x_label, y_label, plot_data1[1], 
                    plot_data1[2], 'MM1', mean_arr2, std_arr2, 
                    'MM2', mean_arr3, std_arr3, 'MM4')


