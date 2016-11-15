import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns

class points_to_complex():
    """decorator for functions that output a list of tuples and
       transform it to a list of complex numbers
    """
    def __init__(self, func):
        self.f = func
    
    def __call__(self, *args):
        return [complex(*point) for point in self.f(*args)] 


class SampleMethods():
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

    def latin_square(self, num_points):
        return self.position_to_points(list(zip(
            np.arange(0, num_points, 1),
            np.random.permutation(np.arange(0, num_points, 1))
        )))
        
    def latin_orthogonal_square(self, num_points, mini_squares_len = 10):
        number_of_squares_x = number_of_squares_y = num_points/mini_squares_len
        x_arr = np.array([np.random.permutation(np.arange(0, mini_squares_len, 1)) 
                         for i in range(number_of_squares_x)])
        y_arr = np.array([np.random.permutation(np.arange(0, mini_squares_len, 1)) 
                         for i in range(number_of_squares_y)])
        positions = []
        for i in range(10):
            for j in range(10):
                pos_real = j + 10*i 
                pos_im = x_arr[i, j] * 10 + y_arr[x_arr[i][j], i]
                positions.append((pos_real, pos_im))

        return self.position_to_points(positions)
    
    def plot_sampling(self, positions, title):
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
    
    @points_to_complex
    def position_to_points(self, positions, random):
        """transform a list with tuples with relative positions to a
           list with absolute positions and adds randomness

        params:
            random: how much randomness to add
        """
        x_factor = (1/self.num_points) * (self.max_real-self.min_real)
        y_factor = (1/self.num_points) * (self.max_im-self.min_im)
        points = [] 
        for position in positions: 
            points.append((x_factor*position[0] + np.random.uniform(-random, random, 1),
                           y_factor*position[1] + np.random.uniform(-random, random, 1)))

        return points

def in_mandelbrot(point, iterations):
    """determines whether point is in mandelbrot set"""
    z = 0
    for i in range(iterations):
        z = z**2 + point
        if abs(z)>2:
            return False

    return True

def plot_mandelbrot():
    real_range = 1000
    im_range = 1000
    grid = np.zeros((real_range, im_range))
    start_point = complex(-2, -2)
    for i in range(im_range):
        if i%50==0:
            print(i)
        for j in range(real_range):
            point = start_point + complex((4/real_range)*j, (4/im_range)*i)
            grid[i,j] = 1 if in_mandelbrot(point, 50000) else 0 

    plt.imshow(grid, cmap='bone', extent= (-2,2,-2,2))
    plt.xlabel("imagenary axis")
    plt.ylabel("real axis")
    plt.title("Mandelbrot set")
    plt.show()

def estimate_area(points, chain_length, total_area):    
    correct = 0    
    for point in points:
        correct += 1 if in_mandelbrot(point, chain_length) else 0
    
    return correct/len(points) * total_area

def batch_sampling_area(batches, samples, chain_length, total_area, sample_func, *args):
    "samples in batches of func(*args) returns mean and std"
    outcomes = []
    for i in range(batches):
        print(i)
        for j in range(samples):
            points = sample_func()
            outcomes.append(estimate_area(points, chain_length, total_area)) 
    
    return np.mean(outcomes), np.std(outcomes)


def main():
    chain_length = 10000
    sample = SampleMethods(10, -2, 2, -2, max_im=2)
    sample_func = sample.uniform_random
    print(batch_sampling_area(100, 1000, 10000, sample.get_area(), sample_func, sample))      
    
    #plot_mandelbrot()
    """
    #comment out decorator for sample.uniform before plotting
    sample =SampleMethods(100, 0, 100, 0, 100)    
    sample.plot_sampling(list(sample.uniform_random()), 
                         "uniform sampling for a 100 by 100 grid")
    """    

if __name__=="__main__":
    main()
