import numpy as np 

class TravellingSalesPerson():
    def __init__(self, file_name):
        self.TSP_dict = {}
        self.city_dict = {}
        with open(file_name, 'r') as f:
            description = True
            for line in f:
                line = line.rstrip('\n')
                if line == 'NODE_COORD_SECTION':
                    description = False
                if description:
                    #set characteristics
                    if '\n' in line:
                        print('nooooo')
                    key, value = line.split(':')
                    self.TSP_dict[key.rstrip(' ')] = value
                elif line != 'EOF' and line != 'NODE_COORD_SECTION':
                    name, coor1, coor2 = line.split(' ') 
                    self.city_dict[name] = (float(coor1), float(coor2))
        if self.TSP_dict['DIMENSION'] == len(self.city_dict.keys()):
            raise ValueError('mismatch dimensions')
        self.distances = np.full((int(self.TSP_dict['DIMENSION']), int(self.TSP_dict['DIMENSION'])),
                                -1, dtype=float)
        self.distance_dict = {k:{} for k in self.city_dict.keys()}
    
    def get_distance(self, city1, city2):
        distance = self.distance_dict[city1].get(city2, -1)
        if distance == -1:
            distance = self.calc_distance(city1, city2)
            self.distance_dict[city1][city2] = self.distance_dict[city2][city1] = distance
        return distance

    def calc_distance(self, city1, city2):
        return np.sqrt((self.city_dict[city1][0]-self.city_dict[city2][0])**2 +
                       (self.city_dict[city1][1]-self.city_dict[city2][1])**2)

    def get_distance_route(self, route):
        """calculate distance of a round trip of all cities
        
        params:
            route: a list featuring all cities once 
        """
        total_distance = 0
        for i in range(len(route)-1):
            total_distance += self.get_distance(route[i], route[i+1])  
        return total_distance + self.get_distance(route[0], route[-1])

if __name__ == '__main__':
    tsp = TravellingSalesPerson('test.tsp.txt')   
    print(tsp.TSP_dict )
    print(tsp.city_dict)    
    print(tsp.distance_dict)
    tsp.get_distance('1', '2')
    print(tsp.distance_dict)
