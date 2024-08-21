#
# This is the definition of a maze navigating agent.
#
import pickle
import math

class Predator:
    """
    This is the maze navigating agent
    """
    def __init__(self, coords):
        """
        Creates new Agent with specified parameters.
        Arguments:
            location:               The agent initial position within maze
        """
        self.coords = coords
        self.old_coords = coords
        self.initial_coords = coords
    

    def get_coords(self):
        return self.coords

    def get_distanceToPreyX(self, prey_coords):
        #distance_to_prey_coords = (abs(self.coords[0] - prey_coords[0]), abs(self.coords[1] - prey_coords[1]))
        #squared_distances = [d**2 for d in distance_to_prey_coords]
        #return math.sqrt(sum(squared_distances))
        print("distância de manhatan:")
        distance = self.coords[0] - prey_coords[0]
        print(distance)
        return distance
    
    def get_distanceToPreyY(self, prey_coords):
        #distance_to_prey_coords = (abs(self.coords[0] - prey_coords[0]), abs(self.coords[1] - prey_coords[1]))
        #squared_distances = [d**2 for d in distance_to_prey_coords]
        #return math.sqrt(sum(squared_distances))
        print("distância de manhatan:")
        distance = self.coords[1] - prey_coords[1]
        print(distance)
        return distance
         
    def get_distanceToPrey(self, prey_coords):
        #distance_to_prey_coords = (abs(self.coords[0] - prey_coords[0]), abs(self.coords[1] - prey_coords[1]))
        #squared_distances = [d**2 for d in distance_to_prey_coords]
        #return math.sqrt(sum(squared_distances))
        print("distância de manhatan:")
        distance = abs(self.coords[0]- prey_coords[0]) + abs(self.coords[1]- prey_coords[1])
        print(distance)
        return distance

    def set_coords(self, x, y):
        self.coords = (x, y)

    def move(self, output):
        most_distant_coord = self.distance_to_prey_coords[0]
        most_distant_coord_pos = 0
        for i in range(1, self.distance_to_prey_coords):
            if self.distance_to_prey_coords[i] >= most_distant_coord:
                most_distant_coord_pos = i
        if self.most_distant_coord_pos[most_distant_coord_pos] >= output:
            self.most_distant_coord_pos[most_distant_coord_pos] -=output

    #new move function to make predator move to the east, west, north or south according to output
    def new_move(self, output):
        '''
        '''
        if 0.2 > output >= 0:
            print("predador não se moveu!")
            print(self.get_coords())
        if 0.4 > output >= 0.2:
            print("predador move-se para norte!")
            self.coords[1] += 1 
        if 0.6 > output >= 0.4:
            print("predador move-se para este!")
            self.coords[0] += 1 
        if 0.8 > output >= 0.6:
            print("predador move-se para sul!")
            self.coords[1] -= 1 
        if 0.1 >= output >= 0.8:
            print("predador move-se para oeste!")
            self.coords[0] -= 1


class Prey:

    def __init__(self,coords):
        self.X = coords[0]
        self.Y = coords[1]
        self.coords = coords
        self.initial_coords = coords
        self.old_coords = coords

    def get_coords(self):
        return self.coords
    
    # ps is supposed to be like this ((4,6)(2,3)(7,1)(9,8))
    def move(self, ps):
        closestdistanceX = ps[0][0] - self.X
        closestdistanceY = ps[0][1] - self.Y
        for p in ps:
            distanceX = p[0] - self.X
            distanceY = p[1] - self.Y
            if abs(distanceX) < abs(closestdistanceX):
                closestdistanceX = distanceX
            if abs(distanceY) < abs(closestdistanceY):
                closestdistanceY = distanceY

        if abs(closestdistanceX) < abs(closestdistanceY):
            if closestdistanceX < 0:
                self.X += 1
                print("the prey moved East")
            if closestdistanceX > 0:
                self.X -= 1
                print("the prey moved West")
        if abs(closestdistanceX) > abs(closestdistanceY):
            if closestdistanceY < 0:
                self.Y += 1
                print("the prey moved North")
            if closestdistanceY > 0:
                self.Y -= 1
                print("the prey moved South")