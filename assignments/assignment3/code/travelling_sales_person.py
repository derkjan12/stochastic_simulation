import numpy as np 

class TravellingSalesPerson():
    def __init__(self, file_name):
        self.TSP_dict = {}
        self.city_dict = {}
        with open(file_name, 'r') as f:
            description = True
            for line in f:
                #print(line)
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
                    elem_li = []
                    elements = line.rstrip(' ').lstrip(' ').split(' ')
                    for count, elem in enumerate(elements):
                        if count == 0:
                            name = elem
                            continue
                        try:
                            float(elem)
                            elem_li.append(elem)
                        except ValueError:
                            pass
                        
                    coor1, coor2 = tuple(elem_li) 
                    self.city_dict[name] = (float(coor1), float(coor2))
        self.TSP_dict['DIMENSION'] = int(self.TSP_dict.get(
            'DIMENSION', len(self.city_dict.keys())
        ))
        if self.TSP_dict['DIMENSION'] != len(self.city_dict.keys()):
            raise ValueError('mismatch dimensions')
        self.distances = np.full((int(self.TSP_dict['DIMENSION']),
                                 int(self.TSP_dict['DIMENSION'])),
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

    def set_route(self, file_name):
        self.route = []
        with open(file_name, 'r') as f:
            description = True
            for line in f:
                line = line.rstrip('\n')
                if line == 'TOUR_SECTION':
                    description = False
                if description:
                    #set characteristics
                    key, value = line.split(':')
                    self.TSP_dict[key.rstrip(' ')] = value
                elif line != 'EOF' and line != 'TOUR_SECTION':
                    if not '-' in line: 
                        self.route.append(str(line).strip(' ')) 

if __name__ == '__main__':
    print("the test tour started")
    tsp = TravellingSalesPerson('test.tsp.txt')   
    #print(tsp.TSP_dict )
    #print(tsp.city_dict)    
    #print('new_route')
    print('the smallest tour started')
    tsp = TravellingSalesPerson('TSP-Configurations/eil51.tsp.txt') 
    tsp.set_route('TSP-Configurations/eil51.opt.tour.txt')
    #print('the optimal route is:')
    #print(tsp.route)
    print("the optimal tour has distance {}".format(tsp.get_distance_route(tsp.route)))
    print('the medium tour started')
    tsp = TravellingSalesPerson('TSP-Configurations/a280.tsp.txt') 
    tsp.set_route('TSP-Configurations/a280.opt.tour.txt')
    print("the optimal tour has distance {}".format(tsp.get_distance_route(tsp.route)))
    print('the large tour has started')
    tsp = TravellingSalesPerson('TSP-Configurations/pcb442.tsp.txt') 
    tsp.set_route('TSP-Configurations/pcb442.opt.tour.txt')
    print("the optimal tour has distance {}".format(tsp.get_distance_route(tsp.route)))


 
