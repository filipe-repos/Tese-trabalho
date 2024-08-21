# The Python standard library import
import os
import shutil
import random
import numpy as np
import math
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
import copy

from PIL import Image, ImageGrab
import pygetwindow as gw
import pickle

from agents import Predator, Prey
import turtle

DIM = 2

DIST = 350
WIDTH = 350
HEIGHT = 350
STEP = 7
N_EVALS = 1

def createpredators_bottom(height, width, n, step):
    return [Predator((-width+ predx * step, -height)) for predx in range(n)]

def createPrey(height, width, preds, step):
    encontrei = False
    while True:
        presa = Prey((random.randint(-width//step, width//step) * step, random.randint(-height//step, height//step) * step))
        if presa.get_coords() not in [pred.get_coords() for pred in preds]:
            return presa

def createPreys(height, width, preds, step, n):
    return [createPrey(height, width, preds, step) for _ in range(n)]

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

#normal distance calculations
def toroidalDistancex(tpred, tprey, width):
        x1 = tpred.position()[0]
        x2 = tprey.position()[0]
        dx = x1 - x2#abs(x1 - x2)

        #if dx was a negative
        d1 = (width*2) - dx
        if d1 > width*2:
            d1-= width
            d1 = -d1
        # Consider toroidal wrapping
        toroidal_dx = min(abs(dx), abs(d1))
        if abs(toroidal_dx) == abs(dx):
            toreturn = dx
        if abs(toroidal_dx) == abs(d1):
            toreturn = d1
        return toreturn

def toroidalDistancey(tpred, tprey, height):
        y1 = tpred.position()[1]
        y2 = tprey.position()[1]
        dy = abs(y1 - y2)

        d1 = (height*2) - dy
        if d1 > height*2:
            d1-= height
            d1 = -d1

        # Consider toroidal wrapping
        toroidal_dy = min(abs(dy), abs(d1))

        if abs(toroidal_dy) == abs(dy):
            toreturn = dy
        if abs(toroidal_dy) == abs(d1):
            toreturn = d1
        return toreturn

def toroidalDistance_coords( pos1, pos2, width, height):
    x1,y1 = pos1
    x2,y2 = pos2
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)

    # Consider toroidal wrapping
    toroidal_dx = min(dx, ((width*2)-dx +1))
    toroidal_dy = min(dy, ((height*2)-dy+1))
    #euclidian distance
    #return (toroidal_dx**2 + toroidal_dy**2)**0.5
    #manhatan distance for ortogonal env
    return abs(toroidal_dx) + abs(toroidal_dy)

def toroidalDistance(tpred, tprey, width, height):
        #calculating the distance using the t_pred pos and t_prey pos
        x1, y1 = tpred.position()
        x2, y2 = tprey.position()
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)

        # Consider toroidal wrapping
        toroidal_dx = min(dx, ((width*2)-dx +1))
        toroidal_dy = min(dy, ((height*2)-dy+1))
        #euclidian distance
        #return (toroidal_dx**2 + toroidal_dy**2)**0.5
        #manhatan distance for ortogonal env
        return abs(toroidal_dx) + abs(toroidal_dy)


def toroidal_pos(pos, max_width, max_height):
    '''
    pega em pos e se sair fora dos limites volta a por dentro dos limites do outro lado(Toroide)
    '''
    x,y = pos
    if abs(x) > max_width:
         x = math.copysign(max_width, x * -1)
    if abs(y) > max_height:
         y = math.copysign(max_height, y * -1)
    return x,y
        


def toroidalcheck(turtle, max_width, max_height, speed):
    if abs(turtle.xcor()) > max_width:
        turtle.hideturtle()
        turtle.penup()
        turtle.speed(max_width)
        turtle.setx(math.copysign(max_width, turtle.xcor() * -1))
        turtle.speed(speed)
        turtle.showturtle()
        turtle.pendown()
    if abs(turtle.ycor()) > max_height:
        turtle.hideturtle()
        turtle.penup()
        turtle.speed(max_height)
        turtle.sety(math.copysign(max_height, turtle.ycor() * -1))
        turtle.speed(speed)
        turtle.showturtle()
        turtle.pendown()

def tpred_move(pred, output, speed):
    pred.old_pos = pred.pos()
    if 0.2 > output >= 0:
        #print("predador não se moveu!")
        return
    elif 0.4 > output >= 0.2:
        #print("predador move-se para este!")
        pred.setheading(0)
    elif 0.6 > output >= 0.4:
        #print("predador move-se para norte!")
        pred.setheading(90)        
    elif 0.8 > output >= 0.6:
        #print("predador move-se para oeste!")
        pred.setheading(180)        
    else:
        #print("predador move-se para sul!")
        pred.setheading(270)       
    pred.forward(speed)
    toroidalcheck(pred, WIDTH, HEIGHT, speed)

def prey_move(prey, tpreds, step):
    closestdistance = toroidalDistance_coords(tpreds[0].old_pos, prey.pos(), WIDTH, HEIGHT)
    closestpred = tpreds[0]
    for tp in tpreds[1:]:
        distance = toroidalDistance_coords(tp.old_pos, prey.pos(), WIDTH, HEIGHT)
        if distance < closestdistance:
            closestdistance = distance
            closestpred = tp
    print("o predador mais próximo: ", closestpred.pos(), "cor: ", closestpred.color())
    #para manter um registo da posição anterior
    prey.old_pos = prey.pos()
    xcoor,ycoor = prey.pos()

    directions = [((step,0), 0), ((-step, 0), 180), ((0, step),90), ((0,-step), 270)]
    farthest = -1
    farthestheading = None
    print("estou em:", xcoor, ycoor)
    random.shuffle(directions)
    for ((ax, ay), a) in directions:
        newpos = ax+xcoor , ay+ ycoor 
        newpos =toroidal_pos(newpos, WIDTH, HEIGHT)
        #toroidalcheck(prey, WIDTH, HEIGHT, step)
        dist =toroidalDistance_coords(closestpred.old_pos, newpos, WIDTH, HEIGHT)
        print("closestpredoldpos:", closestpred.old_pos, "newpreypos:", newpos, "  dist:", dist)
        
        if dist > farthest:
            farthest = dist
            farthestheading = a
            print("melhorei")

    prey.setheading(farthestheading)
    prey.forward(step)
    toroidalcheck(prey, WIDTH, HEIGHT, step)

#def prey_move2(prey,step):
#    prey.setheading(90)
#    prey.forward(200)
#    toroidalcheck(prey, WIDTH, HEIGHT, step)

#captura: a presa é capturada quando estiver na mesma posição de um dos predadores ou
#quando a presa trocou de posição com um dos predadores(trocaram-se)

def captura(preds, prey):
    for p in preds:
        if prey.pos() == p.pos():
            return True
    return atravessaram(preds, prey)

def atravessaram(preds, prey):
    for p in preds:
        if p.old_pos == prey.pos() and prey.old_pos == p.pos():
            return True
    return False    

#preds_list= createpredators_obsolete(DIM, DIST, N_EVALS)
#prey = createPrey_obsolete(DIM, DIST)
preds_def = createpredators_bottom(HEIGHT, WIDTH, 4, STEP)
preys_def = createPreys(HEIGHT, WIDTH, preds_def, STEP, N_EVALS)
preys_test = createPreys(HEIGHT, WIDTH, preds_def, STEP, N_EVALS)

#simula
def simula(net, dim, preds, preys1, preys2):
    cont = 0
    for prey in preys1:
        cont+=1
        print("simulação",cont)
        simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), dim)
    #testing best genome in new situations with new prey positions
    print("testing best genome in new situations with new prey positions")
    for prey in preys2:
        simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), dim)

#dim assumindo que é um quadrado
def simula1(net, preds, prey, dim):
    
    predator1, predator2, predator3, predator4 = preds
    preds_coords = [predator1.get_coords(), predator2.get_coords(), predator3.get_coords(), predator4.get_coords()]
    prey_coords = prey.get_coords()

    map = turtle.Screen()
    map.screensize(dim, dim)
    map.bgcolor("lightgreen")    # set the window background color
    #map.tracer(0, 0)             # to make map not display anything making the execution much faster

    tpred1 = turtle_agent(preds_coords[0], "orange")
    tpred2 = turtle_agent(preds_coords[1], "yellow")
    tpred3 = turtle_agent(preds_coords[2], "red")
    tpred4 = turtle_agent(preds_coords[3], "black")
    tprey = turtle_agent(prey_coords, "blue", "turtle")

    count = 0

    frames = []
    window = gw.getWindowsWithTitle("Python Turtle Graphics")[0]
    com,larg =window.size
    window.moveTo(10,10)

    while count <= ((HEIGHT*2) / STEP) * 1.5: #(500 * 2 / 10) * (3/2) = 150
        count +=1
        
        input1 = toroidalDistancex(tpred1, tprey, WIDTH)/ (WIDTH*2)
        input2 = toroidalDistancey(tpred1, tprey, HEIGHT)/ (HEIGHT*2)
        input3 = toroidalDistancex(tpred2, tprey, WIDTH)/ (WIDTH*2)
        input4 = toroidalDistancey(tpred2, tprey, HEIGHT)/ (HEIGHT*2)
        input5 = toroidalDistancex(tpred3, tprey, WIDTH)/ (WIDTH*2)
        input6 = toroidalDistancey(tpred3, tprey, HEIGHT)/ (HEIGHT*2)
        input7 = toroidalDistancex(tpred4, tprey, WIDTH)/ (WIDTH*2)
        input8 = toroidalDistancey(tpred4, tprey, HEIGHT)/ (HEIGHT*2)
        print("distance in x from pred1 to prey", toroidalDistancex(tpred1, tprey, WIDTH))

        input_data1 = [input1, input2, input3, input4, input5, input6, input7, input8]
        input_data2 = [input3, input4, input1, input2, input5, input6, input7, input8]
        input_data3 = [input5, input6, input1, input2, input3, input4, input7, input8]
        input_data4 = [input7, input8, input1, input2, input3, input4, input5, input6]
        #print("INPUTS!:", input_data1)

        outputp1 = net.activate(input_data1)[0]
        outputp2 = net.activate(input_data2)[0]
        outputp3 = net.activate(input_data3)[0]
        outputp4 = net.activate(input_data4)[0]
        #print("OUTPUTS!:",outputp1, outputp2, outputp3, outputp4)

        tpred_move(tpred1, outputp1, STEP)
        tpred_move(tpred2, outputp2, STEP)
        tpred_move(tpred3, outputp3, STEP)
        tpred_move(tpred4, outputp4, STEP)
        #To make the prey not move commented the method function to make it move
        prey_move(tprey, (tpred1, tpred2, tpred3, tpred4), STEP)
        #prey_move2(tprey, STEP)

        inidist1 = toroidalDistance_coords(tpred1.initial_pos, tprey.position(), HEIGHT, WIDTH)
        inidist2 = toroidalDistance_coords(tpred2.initial_pos, tprey.position(), HEIGHT, WIDTH)
        inidist3 = toroidalDistance_coords(tpred3.initial_pos, tprey.position(), HEIGHT, WIDTH)
        inidist4 = toroidalDistance_coords(tpred4.initial_pos, tprey.position(), HEIGHT, WIDTH)
        mediainidists = (inidist1 + inidist2+ inidist3+ inidist4) / 4
        #print("Media das distâncias ortogonais iniciais de todos os predadores à presa:", mediainidists)
        dist1 = toroidalDistance(tpred1, tprey, HEIGHT, WIDTH)
        dist2 = toroidalDistance(tpred2, tprey, HEIGHT, WIDTH)
        dist3 = toroidalDistance(tpred3, tprey, HEIGHT, WIDTH)
        dist4 = toroidalDistance(tpred4, tprey, HEIGHT, WIDTH)
        mediafinaldists = (dist1 + dist2 + dist3 + dist4)/ 4
        #print("Media das distâncias ortogonais finais de todos os predadores à presa:", mediafinaldists)

        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)

        if captura((tpred1, tpred2, tpred3, tpred4), tprey):
            #print("entrei")
            print("presa apanhada!!!")
            #print("fitness:", 1)
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ 10)
            map.clearscreen()
            frames[0].save("predatorTrialSuccess.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=1)  # Set loop to 0 for infinite loop
            return
    map.clearscreen()
    #print("fitness:",1/(dist1 + dist2 + dist3 + dist4))
    print("fitness:", (mediainidists - mediafinaldists) / 10)
    frames[0].save("best_genomeTrialRun.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=2)  # Set loop to 0 for infinite loop


# the experiment

local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

config_path = os.path.join(local_dir, 'exercise.ini')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


genome_path = 'bestgenome.pkl'
# Later, you can load the genome from the file
with open(genome_path, 'rb') as f:
        #loaded_genome = neat.Genome()
        #loaded_genome.parse(f)
    loaded_genome = pickle.load(f)

preds_def = createpredators_bottom(HEIGHT, WIDTH, 4, STEP)
preys_def = createPreys(HEIGHT, WIDTH, preds_def, STEP, N_EVALS)
preys_test = createPreys(HEIGHT, WIDTH, preds_def, STEP, N_EVALS)

net = neat.nn.FeedForwardNetwork.create(loaded_genome, config)
simula(net, DIST, preds_def, preys_def, preys_test)
print("The END.")