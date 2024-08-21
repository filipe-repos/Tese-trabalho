#
# This is the definition of a maze navigating agent.
#
import pickle
import turtle

class Predator:
    """
    This is the maze navigating agent
    """
    def __init__(self, distance_to_prey1):
        """
        Creates new Agent with specified parameters.
        Arguments:
            location:               The agent initial position within maze
        """
        self.distance_to_prey1 = distance_to_prey1
    
    def get_distanceToPrey(self):
        return self.distance_to_prey1

    def move(self, output):
        #if self.distance_to_prey1 >= output:
        #    self.distance_to_prey1 -=output
        self.distance_to_prey1 -= output
    

def turtle_agent(agent_coords, color= "blue", forma = "circle"):
    ag = turtle.Turtle(shape = forma, visible= False)      # create a turtle named tpred
    ag.color(color)          # make tess blue
    ag.pensize(1)             # set the WIDTH of her pen

    ag.penup()
    ag.goto(agent_coords[0], agent_coords[1])

    ag.showturtle()
    ag.pendown()

    ag.old_pos = None
    ag.initial_pos = ag.position()
    return ag

def turtlemove(turtle, output, step):
    if output >=0.5:
        turtle.forward(step)
