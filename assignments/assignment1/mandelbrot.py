import matplotlib.pyplot as plt
import numpy as np
import time
import json
#import seaborn as sns

class points_to_complex():
    """decorator for functions that output a list of tuples and
       transform it to a list of complex numbers
    """
    def __init__(self, func):
        self.f = func
    
    def __call__(self, *args):
        return [complex(*point) for point in self.f(*args)] 


class Sample():
    def __init__(self, min_real, max_real, min_im, max_im):
        if min_real>max_real or min_im>max_im:
            raise ValueError("min should not be bigger than max")
        self.min_real = min_real
        self.max_real= max_real
        self.min_im= min_im
        self.max_im= max_im
        
    def uniform_random(self, num_points):
        points = zip(np.random.uniform(self.min_real, self.max_real, num_points), 
                     np.random.uniform(self.min_im, self.max_im, num_points))

        return [complex(*point) for point in list(points)] 

    @staticmethod
    def latin_square(num_points):
        return list(zip(
            np.arange(0, num_points, 1),
            np.random.permutation(np.arange(0, num_points, 1))
        ))
        
    @staticmethod
    def latin_orthogonal_square(num_points, mini_squares_len = 10):
        if num_points%mini_squares_len != 0:
            raise ValueError("number_of_points must be a multiple of mini_square length")
                
        num_square_x = num_square_y = num_points//mini_squares_len
       
        first_mat = [] 
        for i in range(num_square_x): 
            first_mat.append(
                np.random.permutation(
                    np.arange(mini_squares_len*i, mini_squares_len*(i+1), 1)
                )
            )
        
        second_mat = [] 
        points_per_bracket = mini_squares_len/num_square_y
        for bracket in range(num_square_y):
            row_second_mat = []
            for row in first_mat:
                row_second_mat.extend(row[int(bracket*points_per_bracket):int((bracket+1)*points_per_bracket)])
            
            second_mat.append(row_second_mat)

        positions = []
        for i, row in enumerate(second_mat):
            li = list(np.random.permutation(np.arange(mini_squares_len*i, mini_squares_len*(i+1), 1)))
            positions.extend(list(zip(li, row)))

        return positions
    
    @staticmethod
    def plot_sampling(positions, title):
        """plots the position given"""
        N = len(positions)        
        image =np.zeros((N, N))
        for position in positions:
            image[position[0], position[1]] = 1
         
        plt.imshow(image, cmap='bone')
        plt.title(title)
        plt.show()
    
    def get_area(self):
        return (self.max_real-self.min_real) * (self.max_im-self.min_im) 
    
    def position_to_points(self, positions, num_points, random=0.5):
        """transform a list with tuples with relative positions to a
           list with absolute positions and adds randomness

        params:
            random: how much randomness to add
        """
        
        #print("random is {}".format(random))
        x_factor = (1/(num_points-1)) * (self.max_real-self.min_real)
        y_factor = (1/(num_points-1)) * (self.max_im-self.min_im)
        points = [] 
        for position in positions: 
            points.append((self.min_real + x_factor* (position[0]+np.random.uniform(-random, random, 1)),
                           self.min_im + y_factor* (position[1]+np.random.uniform(-random, random, 1))))

        return [complex(*point) for point in points]

def in_mandelbrot(point, iterations):
    """determines whether point is in mandelbrot set"""
    z = 0
    for i in range(iterations):
        z = z**2 + point
        if abs(z)>2:
            return False

    return True

def plot_mandelbrot():
    real_range = 800
    im_range = 800
    grid = np.zeros((real_range, im_range))
    start_point = complex(-2, -2)
    for i in range(im_range):
        if i%50==0:
            print(i)
        for j in range(real_range):
            point = start_point + complex((4/real_range)*j, (4/im_range)*i)
            grid[i,j] = 1 if in_mandelbrot(point, 200000) else 0 

    plt.imshow(grid, cmap='bone', extent= (-2,2,-2,2))
    plt.xlabel("real axis")
    plt.ylabel("imagenary axis")
    plt.title("Mandelbrot set")
    plt.show()

def estimate_area(points, chain_length, total_area):    
    correct = 0    
    for point in points:
        correct += 1 if in_mandelbrot(point, chain_length) else 0
    
    return correct/len(points) * total_area

def batch_sampling_area(sample, batch_size, sample_size, chain_length, total_area, sample_func, *args):
    "samples in batches of func(*args) returns mean and std"    
    outcomes = []
    for i in range(batch_size):
        print(i, end=" ", flush=True)
        temp = []
        for j in range(sample_size):
            #points = sample_func(*args)            
            points = sample.position_to_points(sample_func(*args), args[0])
            temp.append(estimate_area(points, chain_length, total_area))
        outcomes.append(temp) 
    
    return outcomes

def compare_areas():
    sample_size, batch_size = 30, 30
    chain_length = 10000
    sample = Sample(-2, 2, -2, max_im=2)
    
    start = time.time()

    area = sample.get_area()    
    mean_li = np.zeros(10)
    std_li = np.zeros(10)
    amount = 100
    #for i in range(10):
    #    mean_li[i], std_li[i] = batch_sampling_area(batch_size, sample_size, amount*(i+1), amount*(i+1), area, sample.uniform_random)    

    with open("stats/uniform_sampling_batch_spec_range", 'w') as f:
        json.dump([list(mean_li), list(std_li)], f)                

    print(batch_sampling_area(batch_size, sample_size, 100000, 100, area, sample.uniform_random))    
    print("time: {}".format(time.time()-start))
    
def chain_length_vs_sample_points():
    sample = Sample(-2, 1, -1, max_im=1)    
    area = sample.get_area()    
    sample_size = 50 
    amount_of_points_li = [10, 50, 100, 500, 1000, 5000, 10000]    
    chain_length_li = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    
    area_estimate = np.zeros((len(amount_of_points_li), len(chain_length_li), 2))
    for i, number_points in enumerate(amount_of_points_li):    
        print(i)
        for j, chain_len in enumerate(chain_length_li):
            temp = batch_sampling_area(1, sample_size, chain_len, area, sample.uniform_random, number_points)[0]
            area_estimate[i, j] = np.mean(temp), np.std(temp)                
    with open("stats/chain_length_vs_num_points.np", 'wb') as f:
        np.save(f, area_estimate)

def main():
    #Sample.plot_sampling(Sample.latin_orthogonal_square(200, 40), "tada")
    
    #chain_length_vs_sample_points()
    
    
    sample_size = 50 
    sample = Sample(-2, 1, -1, 1)
    chain_len = 10000
    number_points_li = [500, 1000, 2000, 4000, 8000]
    result_li = []    
    for number_points in number_points_li:    
        result_li.append(batch_sampling_area(
            sample, 1, sample_size, chain_len, sample.get_area(), 
            sample.latin_orthogonal_square, number_points, number_points//10
        )[0])

    print(result_li)
    res = [list(np.mean(result_li, axis=1)), list(np.std(result_li, axis=1))]
    with open('sampling_comparison_ortho.json', 'w') as f:
        json.dump(res, f)
            

    """
    #comment out decorator for sample.uniform before plotting
    sample =Sample(100, 0, 100, 0, 100)    
    sample.plot_sampling(list(sample.uniform_random()), 
                         "uniform sampling for a 100 by 100 grid")
    """    

if __name__=="__main__":
    main()
