from agents import Agent
import random
import turtle

#width of map
WIDTH = 350
#height of map
HEIGHT = 350
#Step how much the agents(preds and prey) move per turn. Should allways be DIST/50
STEP = 7

def createPreys9(height, width, step):
    segmentw = width*2/3
    segmenth = height*2/3
    prey1 = Agent((random.randint(-width//step, (-width+ segmentw)//step) * step, random.randint(-height//step, (-height+segmenth)//step) * step))
    prey2 = Agent((random.randint((-width+ segmentw)//step, (width - segmentw)//step) * step, random.randint((-height//step), (-height+segmenth)//step) * step))
    prey3 = Agent((random.randint((width - segmentw)//step, width//step) * step, random.randint(-height//step, (-height+segmenth)//step) * step))
    prey4 = Agent((random.randint(-width//step, (-width+ segmentw)//step) * step, random.randint((-height+segmenth)//step, (height-segmenth)//step) * step))
    prey5 = Agent((random.randint((-width+ segmentw)//step, (width - segmentw)//step) * step, random.randint((-height+segmenth)//step, (height-segmenth)//step) * step))
    prey6 = Agent((random.randint((width - segmentw)//step, width//step) * step, random.randint((-height+segmenth)//step, (height-segmenth)//step) * step))
    prey7 = Agent((random.randint(-width//step, (-width+ segmentw)//step) * step, random.randint((height-segmenth)//step, height//step) * step))
    prey8 = Agent((random.randint((-width+ segmentw)//step, (width - segmentw)//step) * step, random.randint((height-segmenth)//step, height//step) * step))
    prey9 = Agent((random.randint((width - segmentw)//step, width//step) * step, random.randint((height-segmenth)//step, height//step) * step))
    return [prey1, prey2, prey3, prey4, prey5, prey6, prey7, prey8, prey9]

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
    print("x,y: ",ag.pos())
    return ag

wn = turtle.Screen()        # creates a graphics window
wn.screensize(WIDTH, HEIGHT)
wn.bgcolor("white")    # set the window background color

preys = createPreys9(HEIGHT, WIDTH, STEP)

tpreys = []
for prey in preys:
    tprey=turtle_agent(prey.get_coords(), "blue", "turtle")
    tpreys.append(tprey)

wn.exitonclick()