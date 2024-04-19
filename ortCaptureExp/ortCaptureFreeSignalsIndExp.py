#
# This file provides source code of the predator prey coordenation experiment using on NEAT-Python library
#

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

from agents import Agent
import turtle


#DIST is The dimensions of map, onlyapplicable if map is square
DIST = 350
#width of map
WIDTH = 350
#height of map
HEIGHT = 350
#Step how much the agents(preds and prey) move per turn. Should allways be DIST/50
STEP = 7
#N_Evals is the number of different evaluations(cases where prey is in different position) done per genome per generation
N_EVALS = 1
#N_PREDS is the number of predators to be present in experiment and to chase the prey
N_PREDS = 3
#TICKS is the limit of turns allowed or movements to be done by all agents before the experiment ends
TICKS = int(((HEIGHT*2) / STEP) * 1.5)

#older version that generates quadpairs of preds in random positions not to use
def createpredators_obsolete(dim, dist, nquadpairpredators):
    pred_list = []
    for pairpredators in range(nquadpairpredators):
        random_pred1 = tuple(random.randint(-dist//STEP, dist//STEP)* STEP for n in range(dim))
        random_pred2 = tuple(random.randint(-dist, dist) for n in range(dim))
        random_pred3 = tuple(random.randint(-dist, dist) for n in range(dim))
        random_pred4 = tuple(random.randint(-dist, dist) for n in range(dim))
        pred1 = Agent(random_pred1)
        pred2 = Agent(random_pred2)
        pred3 = Agent(random_pred3)
        pred4 = Agent(random_pred4)
        pred_list.append((pred1,pred2,pred3,pred4))
    return pred_list

#older version not to use
def createPrey_obsolete(dim,dist):
    random_prey = tuple(random.randint(-dist, dist) for n in range(dim))
    return Agent(random_prey)

def createpredators_bottom(height, width, n, step):
    return [Agent((-width+ predx * step, -height)) for predx in range(n)]

def createpredators_right(height, width, n, step):
    return [Agent((width - predx * step, height)) for predx in range(n)]

def createPrey(height, width, preds, step):
    encontrei = False
    while True:
        presa = Agent((random.randint(-width//step, width//step) * step, random.randint(-height//step, height//step) * step))
        if presa.get_coords() not in [pred.get_coords() for pred in preds]:
            return presa

def createPreys(height, width, preds, step, n):
    return [createPrey(height, width, preds, step) for _ in range(n)]

def createPreys9(height, width, preds, step):
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
    return ag

#change this function bellow it isnt optimal follow example from toroidalDistance_coords
#normal distance calculations

#functions used for agents that are not represented in turtles environment
def toroidal_distance_d(coord1, coord2, max_d): 
    d = abs(coord1 - coord2)
    # Consider toroidal wrapping
    toroidal_d = min(d, ((max_d*2)-d +1))
    return toroidal_d

#distance between to coords, offsets are negative se a presa estiver em cima ou à direita do predador, e positivos se em baixo ou à esquerda do predador
def toroidalDistance1coord(coord1, coord2, dim):
    dif = coord1 - coord2
    if dif < 0:
        if abs(dif) <= dim:
            return dif
        return  2*dim - abs(dif)
    if dif <= dim:
        return dif
    return -(2*dim - dif)

#print(toroidalDistance1coord(-350, -170, 350))
#input()

#undone maybe a new better version of toroidalDistance1coord()
def toroidalDistance_coord(c1, c2, dim):
        dy = c1 - c2#abs(y1 - y2)

        d1 = dim - dy
        if d1 > dim:
            d1-= dim
            d1 = -d1

        # Consider toroidal wrapping
        toroidal_dy = min(abs(dy), abs(d1))

        if abs(toroidal_dy) == abs(dy):
            toreturn = dy
        if abs(toroidal_dy) == abs(d1):
            toreturn = d1
        return toreturn

#calculates toroidal distance given 2 positions in 2d space
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

#calculates toroidal distance given 2 turtle objects
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

#only to check and update position of agents according to toroidal environment, no visualization
def toroidal_pos(pos, max_width, max_height):
    '''
    pega em pos e se sair fora dos limites volta a por dentro dos limites do outro lado(Toroide)
    '''
    x,y = pos
    if abs(x) > max_width:
        x = math.copysign(max_width-(abs(x) - max_width), x * -1)
    if abs(y) > max_height:
        y = math.copysign(max_height-(abs(y) - max_height), y * -1)
    return x,y

#to use only in simulation with visualization        
def toroidalcheck(turtle, max_width, max_height, speed):
    if abs(turtle.xcor()) > max_width:
        turtle.hideturtle()
        turtle.penup()
        turtle.speed(max_width)
        turtle.setx(math.copysign(max_width-(abs(turtle.xcor()) - max_width), turtle.xcor() * -1))
        turtle.speed(speed)
        turtle.showturtle()
        turtle.pendown()
    if abs(turtle.ycor()) > max_height:
        turtle.hideturtle()
        turtle.penup()
        turtle.speed(max_height)
        turtle.sety(math.copysign(max_height-(abs(turtle.ycor()) - max_height), turtle.ycor() * -1))
        turtle.speed(speed)
        turtle.showturtle()
        turtle.pendown()

# to move turtle object pred 
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

# to move not turtle object pred        
def pred_move(pred, output, step):
    x,y = pred.get_coords()
    pred.old_coords = pred.get_coords()
    if 0.2 > output >= 0:
        #print("predador não se moveu!")
        return
    elif 0.4 > output >= 0.2:
        #print("predador move-se para este!")
        pred.set_coords(x+step, y)
    elif 0.6 > output >= 0.4:
        #print("predador move-se para norte!")
        pred.set_coords(x, y+step)        
    elif 0.8 > output >= 0.6:
        #print("predador move-se para oeste!")
        pred.set_coords(x-step, y)        
    else:
        #print("predador move-se para sul!")
        pred.set_coords(x, y-step)
    new_coords = pred.get_coords()         
    x,y =toroidal_pos(new_coords, WIDTH, HEIGHT)
    pred.set_coords(x,y)

#to move turtle object prey
def tprey_move(prey, tpreds, step):
    #calculating closest pred
    closestdistance = toroidalDistance_coords(tpreds[0].old_pos, prey.pos(), WIDTH, HEIGHT)
    closestpred = tpreds[0]
    for tp in tpreds[1:]:
        distance = toroidalDistance_coords(tp.old_pos, prey.pos(), WIDTH, HEIGHT)
        if distance < closestdistance:
            closestdistance = distance
            closestpred = tp
    #print("o predador mais próximo: ", closestpred.pos(), "cor: ", closestpred.color())
            
    #para manter um registo da posição anterior
    prey.old_pos = prey.pos()
    xcoor,ycoor = prey.pos()

    directions = [((step,0), 0), ((-step, 0), 180), ((0, step),90), ((0,-step), 270)]
    farthest = -1
    farthestheading = None
    #print("estou em:", xcoor, ycoor)
    random.shuffle(directions)
    for ((ax, ay), a) in directions:
        newpos = ax+xcoor , ay+ ycoor 
        newpos =toroidal_pos(newpos, WIDTH, HEIGHT)
        dist =toroidalDistance_coords(closestpred.old_pos, newpos, WIDTH, HEIGHT)
        #print("closestpredoldpos:", closestpred.old_pos, "newpreypos:", newpos, "  dist:", dist)
        
        if dist > farthest:
            farthest = dist
            farthestheading = a
            #print("melhorei")

    prey.setheading(farthestheading)
    prey.forward(step)
    toroidalcheck(prey, WIDTH, HEIGHT, step)           


#to move not turtle object prey 
def prey_move(prey, preds, step):
    #calculating closest pred
    closestdistance = toroidalDistance_coords(preds[0].old_coords, prey.get_coords(), WIDTH, HEIGHT)
    closestpred = preds[0]
    for tp in preds[1:]:
        distance = toroidalDistance_coords(tp.old_coords, prey.get_coords(), WIDTH, HEIGHT)
        if distance < closestdistance:
            closestdistance = distance
            closestpred = tp
    #print("o predador mais próximo: ", closestpred.pos(), "cor: ", closestpred.color())
            
    #para manter um registo da posição anterior
    prey.old_coords = prey.get_coords()
    xcoor,ycoor = prey.get_coords()

    directions = [(step,0), (-step, 0), (0, step), (0,-step)]
    farthest = -1
    farthestheading = None
    #print("estou em:", xcoor, ycoor)
    random.shuffle(directions)
    for (ax, ay) in directions:
        newpos = ax+ xcoor , ay+ ycoor 
        newpos = toroidal_pos(newpos, WIDTH, HEIGHT)
        dist =toroidalDistance_coords(closestpred.old_coords, newpos, WIDTH, HEIGHT)
        #print("closestpredoldpos:", closestpred.old_pos, "newpreypos:", newpos, "  dist:", dist)
        
        if dist > farthest:
            x_tomove, y_tomove = newpos
            farthest = dist
            #print("melhorei")

    prey.set_coords(x_tomove, y_tomove)

def new_prey_move(prey, preds, step):
    #calculating closest pred
    pred0x, pred0y=preds[0].old_coords
    preyx, preyy = prey.get_coords()
    closestdistancex = toroidalDistance1coord(pred0x, preyx, WIDTH)
    closestdistancey = toroidalDistance1coord(pred0y, preyy, HEIGHT)
    for tp in preds[1:]:
        tpx,tpy = tp.old_coords 
        distancex = toroidalDistance1coord(tpx, preyx, WIDTH)
        distancey = toroidalDistance1coord(tpy, preyy, WIDTH)
        if abs(distancex) < abs(closestdistancex):
            closestdistancex = distancex
        if abs(distancey) < abs(closestdistancey):
            closestdistancey = distancey
    #print("o predador mais próximo: ", closestpred.pos(), "cor: ", closestpred.color())
            
    #para manter um registo da posição anterior
    prey.old_coords = prey.get_coords()
    xcoor,ycoor = prey.get_coords()

    newpos = xcoor, ycoor
    if abs(closestdistancex) < abs(closestdistancey):
        if closestdistancex < 0:
            if abs(closestdistancex) <= WIDTH:
                newpos =  xcoor + step, ycoor   #direita ou leste
            else:
                newpos = xcoor - step, ycoor    #esquerda ou oeste
        else:
            if abs(closestdistancex) <= WIDTH:
                newpos = xcoor - step, ycoor    #esquerda ou oeste
            else:
                newpos =  xcoor + step, ycoor   #direita ou leste
    else:
        if closestdistancey < 0:
            if abs(closestdistancey) <=HEIGHT:
                newpos =  xcoor, ycoor + step   #cima ou norte
            else:
                newpos = xcoor, ycoor - step    #baixo ou sul
        else:
            if abs(closestdistancey) <=HEIGHT:
                newpos = xcoor, ycoor - step    #baixo ou sul
            else:
                newpos =  xcoor, ycoor + step   #cima ou norte
        
    newpos = toroidal_pos(newpos, WIDTH, HEIGHT)
    x_tomove, y_tomove = newpos
    prey.set_coords(x_tomove, y_tomove)

#to move turtle object prey
def new_tprey_move(prey, tpreds, step):
    #calculating closest pred
    tpredx, tpredy = tpreds[0].old_pos
    preyx, preyy = prey.pos()
    closestdistancex = toroidalDistance1coord(tpredx, preyx, WIDTH)
    closestdistancey = toroidalDistance1coord(tpredy, preyy, HEIGHT)
    closestpred = tpreds[0]
    for tp in tpreds[1:]:
        tpx, tpy = tp.old_pos
        distancex = toroidalDistance1coord(tpx, preyx, WIDTH)
        distancey = toroidalDistance1coord(tpy, preyy, WIDTH)
        if abs(distancex) < abs(closestdistancex):
            closestdistancex = distancex
        if abs(distancey) < abs(closestdistancey):
            closestdistancey = distancey
    #print("o predador mais próximo: ", closestpred.pos(), "cor: ", closestpred.color())
            
    #para manter um registo da posição anterior
    prey.old_pos = prey.pos()
    xcoor,ycoor = prey.pos()

    predx,predy = closestpred.pos()

    dx = toroidalDistance1coord(predx, xcoor, WIDTH)
    dy = toroidalDistance1coord(predy, ycoor, HEIGHT)
    best_heading = 0

    if abs(closestdistancex) < abs(closestdistancey): #distância entre predador mais próximo e presa em x(horizontal) é menor(mais próximo) do que em y(vertical)
        if closestdistancex < 0:
            if abs(dx) <= WIDTH:
                best_heading = 0    #direita ou leste
            else:
                best_heading = 180
        else:
            if abs(dx) <=WIDTH:
                best_heading = 180  #esquerda ou oeste
            else:
                best_heading = 0
    else:   #distância entre predador mais próximo e presa em y(vertical) é menor(mais próximo) do que em x(horizontal)
        if dy < 0:
            if abs(dy) <=HEIGHT:
                best_heading = 90   #cima ou norte
            else:
                best_heading = 270
        else:
            if abs(dy) <=HEIGHT:
                best_heading = 270  #baixo ou sul
            else:
                best_heading = 90
        
    prey.setheading(best_heading)
    prey.forward(step)
    toroidalcheck(prey, WIDTH, HEIGHT, step)


#captura: a presa é capturada quando estiver na mesma posição de um dos predadores ou
#quando a presa trocou de posição com um dos predadores(trocaram-se)

def captura(preds, prey):
    for p in preds:
        if prey.pos() == p.pos():
            return True
    return atravessaram(preds, prey)

def atravessaram(preds, prey):
    for p in preds:
        if p.old_pos == prey.pos() or prey.old_pos == p.pos(): # previous was a and condition
            return True
        #following condition might be wrong used for cases where the preds move faster than prey
        #if prey.pos()[0] in range(p.old_pos[0], p.pos()[0]) or prey.pos()[1] in range(p.old_pos[1], p.pos()[1]): #doesnt work for floats
        if p.old_pos[0] < prey.pos()[0] < p.pos()[0] and p.old_pos[1] < prey.pos()[1] < p.pos()[1]:
            return True
    return False    

#nova versão para objetos não turtle para não visualização
def captura_a(preds, prey):
    for p in preds:
        if prey.get_coords() == p.get_coords():
            return True
    return atravessaram_a(preds, prey)

def atravessaram_a(preds, prey):
    for p in preds:
        if p.get_old_coords() == prey.get_coords() or prey.get_old_coords() == p.get_coords(): # previous was a and condition
            return True
        #following condition might be wrong used for cases where the preds move faster than prey
        #if prey.get_coords()[0] in range(p.get_old_coords()[0], p.get_coords()[0]) or prey.get_coords()[1] in range(p.get_old_coords()[1], p.get_coords()[1]): #doesnt work for floats
        if p.get_old_coords()[0] < prey.get_coords()[0] < p.get_coords()[0] and p.get_old_coords()[1] < prey.get_coords()[1] < p.get_coords()[1]:
            return True
    return False    

#new functions to handle communication
def ann_inputs_outputs_signal(pred, signals, prey, net):
    #signal1, signal2 = signals #mudar porque o numero de sinais depende do numero de predadores e não serão sempre 2
    x,y = pred.get_coords()
    preyx, preyy = prey.get_coords()
    offsetx = toroidalDistance1coord(x, preyx, WIDTH)
    offsety = toroidalDistance1coord(y, preyy, HEIGHT)
    norm_offsetx = ((offsetx/ WIDTH) +1 )/2
    norm_offsety = ((offsety/ HEIGHT) +1 )/2
    input_data = [norm_offsetx, norm_offsety] + signals
    #print("tuple(net.activate(input_data)):", tuple(net.activate(input_data)))
    output, signal = tuple(net.activate(input_data))
    return output, signal

def ann_inputs_outputs_signal_t(tpred, signals, tprey, net):
    x,y = tpred.pos()
    preyx, preyy = tprey.pos()
    offsetx = toroidalDistance1coord(x, preyx, WIDTH)
    offsety = toroidalDistance1coord(y, preyy, HEIGHT)
    norm_offsetx = ((offsetx/ WIDTH) +1 )/2
    norm_offsety = ((offsety/ HEIGHT) +1 )/2
    input_data = [norm_offsetx, norm_offsety] + signals
    output, signal = tuple(net.activate(input_data))
    return output, signal

#simula
def simula(net, preds, preys, preys_test, height, width, ticks):
    cont = 0
    for prey in preys:
        cont+=1
        print("simulação",cont)
        simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), height, width, ticks)
    #testing best genome in new situations with new prey positions
    print("testing best genome in new situations with new prey positions")
    #for prey in preys_test:
    #    simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), height,width, ticks)

def simula1(net, preds, prey, height, width, ticks):
    
    preds = copy.deepcopy(preds)#preds
    prey = copy.deepcopy(prey)
    prey_coords = prey.get_coords()

    colors = ["yellow", "orange", "red", "black", "yellow", "orange", "red", "black"]
    n_preds = len(preds)
    signals = [0 for n in range(n_preds)]

    map = turtle.Screen()
    map.screensize(height, width)
    map.bgcolor("lightgreen")    # set the window background color
    #map.tracer(0, 0)             # to make map not display anything making the execution much faster
    tpreds = []    

    for pred, color in zip(preds, colors):
        tpred = turtle_agent(pred.get_coords(), color)
        tpreds.append(tpred)
        #print("tpred pos: ", tpred.pos())

    tprey = turtle_agent(prey_coords, "blue", "turtle")
    #print("tprey pros: ", tprey.pos())

    frames = []
    window = gw.getWindowsWithTitle("Python Turtle Graphics")[0]
    com,larg =window.size
    window.moveTo(10,10)

    #while count <= ((HEIGHT*2) / STEP) * 1.5: #(500 * 2 / 10) * (3/2) = 150 (350 *2) /7)) *1.5
    for count in range(ticks):
        count +=1
        
        for npredsig in range(n_preds):#for pred, signal in zip (preds, signals):
            tpred = tpreds[npredsig]
            sinaisOutrem =  copy.deepcopy(signals)
            del sinaisOutrem[npredsig]
            output, signal = ann_inputs_outputs_signal_t(tpred, sinaisOutrem, tprey, net)
            signals[npredsig] = signal
            tpred_move(tpred, output, STEP)
            #print("tpred pos:", tpred.pos())

        #print("pred new coords: ", tpred.position())
        #To make the prey not move commented the method function to make it move
        tprey_move(tprey, tpreds, STEP)

        #to delete just to test something
        #print("prey pos:", tprey.pos())

        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)

        if captura(tpreds, tprey):
            print("presa apanhada!!!")

            finaldists = [toroidalDistance_coords(tpred.position(), tprey.position(), HEIGHT, WIDTH) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds

            #print("fitness:", 1)
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ 10)
            map.clearscreen()

            frames[0].save("predatorTrialSuccess.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=1)  # Set loop to 0 for infinite loop
            return

    inidists = [toroidalDistance_coords(tpred.initial_pos, tprey.position(), HEIGHT, WIDTH) for tpred in tpreds]
    mediainidists = sum(inidists) / n_preds

    finaldists = [toroidalDistance_coords(tpred.position(), tprey.position(), HEIGHT, WIDTH) for pred in preds]
    mediafinaldists = sum(finaldists) / n_preds
    
    map.clearscreen()
    #print("fitness:",1/(dist1 + dist2 + dist3 + dist4))
    print("fitness:", (mediainidists - mediafinaldists) / 10)
    frames[0].save("best_genomeTrialRun.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=2)  # Set loop to 0 for infinite loop


#calculo da média dos fitness das várias avaliações de cada rede neuronal numa lista de preys
def eval_fitness(net, preds_def, preys_def, height, width, ticks):
    the_fitness= 0
    #cycle_count = 0
    #print([p.get_coords() for p in preys_def])
    for prey in preys_def:
        #cycle_count += 1
        #print("CYCLE:", cycle_count)
        f1 = eval_fitness1(net, preds_def, prey, height, width, ticks)
        #print("eval1", f1)
        the_fitness += f1
    #print("the summed fitness:", the_fitness, "the number of experiments per genome:", len(preys_def))
    return the_fitness/ len(preys_def)


#Urgent to change 
def eval_fitness1(net, preds_def, theprey, height, width, ticks):
    """
    Evaluates fitness of the genome that was used to generate 
    provided net
    Arguments:
        net: The feed-forward neural network generated from genome
    Returns:
        The fitness score - the higher score the means the better 
        fit organism. Maximal score: 16.0
    """
    the_fitness = 0.0

    preds = copy.deepcopy(preds_def)
    prey = copy.deepcopy(theprey)
    n_preds = len(preds)
    
    #for pred in preds:
        #print("pred pos: ", pred.get_coords())
    #print("prey pos:", prey.get_coords())
    signals = [0 for n in range(n_preds)]

    for count in range(ticks): #(500 * 2 / 10) * (3/2) = 150
        
        for npredsig in range(n_preds):#for pred, signal in zip (preds, signals):
            pred = preds[npredsig]
            sinaisOutrem =  copy.deepcopy(signals)
            del sinaisOutrem[npredsig]
            output, signal = ann_inputs_outputs_signal(pred, sinaisOutrem, prey, net)
            signals[npredsig] = signal
            #no simula por aqui para mudar a cor do agente de acordo com o sinal
            pred_move(pred, output, STEP)

        #To make the prey not move commented the method function to make it move
        prey_move(prey, preds, STEP)

        #print("pred1_initialpos: ", pred1.get_initial_coords())
        #print("prey pos: ", prey.get_coords())

        #print("Media das distâncias ortogonais iniciais de todos os predadores à presa:", mediainidists)
        

        if captura_a(preds, prey):#if dist1 <= 40 or dist2 <= 40 or dist3 <= 40 or dist4 <= 40:

            finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), height, width) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds
        
        #print("Media das distâncias ortogonais finais de todos os predadores à presa:", mediafinaldists)

            print("presa apanhada!!!")
            #print("presa: ", prey.get_coords())
            print()
            print("fitness:", (2*(width + height) - mediafinaldists)/ 10)
            return ((2*(width + height) - mediafinaldists)/ 10) # max threshold is 160 ((1600 - 0) / 10)

    #new code to avoid bad genomes or neural networks that make preds only move in one direction the whole time or all preds move same direction the whole time
    predsposx = []
    predsposy = []
    onedirectionpreds = []
    for pred in preds:
        x,y = pred.get_coords()
        x_initial, y_initial = pred.initial_coords
        predsposx.append(x)
        predsposy.append(y)
        if x == x_initial or y == y_initial:
            onedirectionpreds.append(True)
        else:
            onedirectionpreds.append(False)
    equalsx = all(i == predsposx[0] for i in predsposx)
    equalsy = all(i == predsposy[0] for i in predsposy)
    if all(i == True for i in onedirectionpreds):
        return 1 #bad fitness if all preds only moved in one direction during evolution
    if equalsx or equalsy:
        return 1 #bad fitness if not capture and pos of preds in x or y are equal

    inidists = [toroidalDistance_coords(pred.get_initial_coords(), prey.get_coords(), height, width) for pred in preds]
    mediainidists = sum(inidists) / n_preds

    finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), height, width) for pred in preds]
    mediafinaldists = sum(finaldists) / n_preds
    #print("fitness:",1/(dist1 + dist2 + dist3 + dist4))
    #print("fitness:", (mediainidists - mediafinaldists) / 10)
    return (mediainidists - mediafinaldists) / 10


# more constants

#the list of Preds to be used in in each avaliation
PREDS_DEF = createpredators_bottom(HEIGHT, WIDTH, N_PREDS, STEP)
#the list of Preys to be used in in evalfitness_1()
PREYS_DEF = createPreys(HEIGHT, WIDTH, PREDS_DEF, STEP, N_EVALS)
#the list of Preysc for testing to be used in each simulation1()
PREYS_TEST = createPreys(HEIGHT, WIDTH, PREDS_DEF, STEP, N_EVALS*10)
PREYS_9 = createPreys9(HEIGHT, WIDTH, PREDS_DEF, STEP)
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')


def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    XOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    genome_count = 0
    for genome_id, genome in genomes:
        genome_count += 1
        print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net, PREDS_DEF, PREYS_DEF, HEIGHT, WIDTH, TICKS)

def run_experiment(config_file):
    """
    The function to run XOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(out_dir, 'neat-checkpoint-')))


    # Run for up to 300 generations.
    best_genome = p.run(eval_genomes, 300)#500

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'offx2', -4: 'offy2', -5: 'offx3', -6: 'offy3', -7: 'offx4', -8: 'offy4', 0:'Move_outputp', 1:'Signal'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    print("AQUI!!!")
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))

    print("ALI!!!")
    #save image of plot of the fitness
    #plot1 = gw.getWindowsWithTitle("Figure 1")[0]
    #com,larg =plot1.size
    #plot1.moveTo(10,10)
    #image1 = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
    #image1.save("results\ortogonalPredatorsProblemResultsFitness.png")
    print("Acoli!!!")

    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

    #save image of plot of the species
    #plot2 = gw.getWindowsWithTitle("Figure 1")[0]
    #com2,larg2 =plot2.size
    #plot2.moveTo(500, 500)
    #image2 = ImageGrab.grab(bbox=(500, 500, 500+com2, 500+larg2))
    #image2.save("results\ortogonalPredatorsProblemResultsSpecies.png")

    return best_genome


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)

def nrunexperiment(n):
    best_of_the_bestGenome = None
    best_of_the_bestGenomefitness = 0
    #run this n times
    for i in range(n):
        if __name__ == '__main__':
            # Determine path to configuration file. This path manipulation is
            # here so that the script will run successfully regardless of the
            # current working directory.
            config_path = os.path.join(local_dir, 'exercise.ini')

            # Clean results of previous run if any or init the output directory
            
            clean_output()
            print("BEGINNING!")

            # Run the experiment
            best_genome = run_experiment(config_path)
            print("best_genome.fitness:", best_genome.fitness)
            #to get the best genome out of the n resulting best genomes of the n experiments 
            if best_genome.fitness > best_of_the_bestGenomefitness:
                best_of_the_bestGenomefitness = best_genome.fitness
                best_of_the_bestGenome = best_genome
                
            # Assuming 'genome' is your NEAT genome object
            genome_path = 'goodgenomes.pkl'

            # Save the genome to a file
            with open("goodgenomes.pkl", "wb") as f:
                pickle.dump(best_genome, f)
                f.close()

            if i < n-1:
                userinput = str(input("want to carry on with the program?:(y/n) "))
                if userinput == "n":
                    break
    #keep the best genome of the n experimentations in a separate file
    best_genome_path = 'bestgenome.pkl'
    with open("bestgenome.pkl", "wb") as f:
            pickle.dump(best_of_the_bestGenome, f)
            f.close()
    print("best_of_the_bestGenome.fitness", best_of_the_bestGenome.fitness)
    print("end of regular experimentation!")
    print()

    print("simulate behavior of best genome:", best_of_the_bestGenome)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    net = neat.nn.FeedForwardNetwork.create(best_of_the_bestGenome, config)
    simula(net, PREDS_DEF, PREYS_DEF, PREYS_TEST, HEIGHT, WIDTH, TICKS)
    print("The END.")


nrunexperiment(5)