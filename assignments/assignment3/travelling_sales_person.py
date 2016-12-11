import numpy 

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
                    print(line.split(':'))
                    self.TSP_dict.update([line.split(':')])
                elif line != 'EOF' and line != 'NODE_COORD_SECTION':
                    name, coor1, coor2 = line.split(' ') 
                    self.city_dict[name] = (coor1, coor2)

print('hello')
if __name__ == '__main__':
    print('hello')
    tsp = TravellingSalesPerson('test.tsp.txt')   
    print(tsp.TSP_dict )
    
