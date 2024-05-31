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

#File containing all functions used by the capture experience

#DIST is The dimensions of map, onlyapplicable if map is square
DIST = 350
#width of map
WIDTH = 350
#height of map
HEIGHT = 350
#Step how much the agents(preds and prey) move per turn. Should allways be DIST/50
STEP = 7


def createpredators_bottom(height, width, n, step):
    return [Agent((-width+ predx * step, -height)) for predx in range(n)]

def createpredators_right(height, width, n, step):
    return [Agent((width - predx * step, height)) for predx in range(n)]

def createpredators_edges(height, width, n, step):
    pass


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
    return (toroidal_dx**2 + toroidal_dy**2)**0.5
    #manhatan distance for ortogonal env
    #return abs(toroidal_dx) + abs(toroidal_dy)

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
    #other way
    #if x > max_width:
    #    dif = x - max_width
    #    x = -max_width + dif
    #elif x < -max_width:
    #    x = max_width + (x + max_width) 
    #if y > max_height:
    #    dif = y - max_height
    #    y = -max_height + dif
    #elif y < -max_height:
    #    y = max_height + (y + max_height) 
    #return x,y

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
    #to make the prey move only when a predator is close enough to it 
    if closestdistance > 10*STEP:
        return
    
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
    #to make the prey move only when a predator is close enough to it 
    if closestdistance > 10*STEP:
        return
    
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

# to move turtle object pred with 5 outputs 
def tpred_move5(tpred, outputs, speed):
    tpred.old_pos = tpred.pos()
    biggerOutput = outputs[0]
    biggerOutputN = 0
    for n in range(1, len(outputs)):
        if outputs[n] > biggerOutput:
            biggerOutput = outputs[n]
            biggerOutputN = n
    if biggerOutputN == 0:
        #print("predador não se moveu!")
        return
    elif biggerOutputN == 1:
        #print("predador move-se para este!")
        tpred.setheading(0)
    elif biggerOutputN == 2:
        #print("predador move-se para norte!")
        tpred.setheading(90)        
    elif biggerOutputN == 3:
        #print("predador move-se para oeste!")
        tpred.setheading(180)        
    else:
        #print("predador move-se para sul!")
        tpred.setheading(270)       
    tpred.forward(speed)
    toroidalcheck(tpred, WIDTH, HEIGHT, speed)

# to move not turtle object pred with 5 outputs       
def pred_move5(pred, outputs, step):
    x,y = pred.get_coords()
    pred.old_coords = pred.get_coords()
    biggerOutput = outputs[0]
    biggerOutputN = 0
    for n in range(1, len(outputs)):
        if outputs[n] > biggerOutput:
            biggerOutput = outputs[n]
            biggerOutputN = n
    if biggerOutputN == 0:
        #print("predador não se moveu!")
        return
    elif biggerOutputN == 1:
        #print("predador move-se para este!")
        pred.set_coords(x+step, y)
    elif biggerOutputN == 2:
        #print("predador move-se para norte!")
        pred.set_coords(x, y+step)        
    elif biggerOutputN == 3:
        #print("predador move-se para oeste!")
        pred.set_coords(x-step, y)        
    else:
        #print("predador move-se para sul!")
        pred.set_coords(x, y-step)
    new_coords = pred.get_coords()         
    x,y =toroidal_pos(new_coords, WIDTH, HEIGHT)
    #print("pred x,y: ", x,y)
    pred.set_coords(x,y)

#captura: a presa é capturada quando estiver na mesma posição de um dos predadores ou
#quando a presa trocou de posição com um dos predadores(trocaram-se)
def captura(preds, prey):
    for p in preds:
        if prey.pos() == p.pos():
            return True
    return atravessaram(preds, prey)

def atravessaram(preds, prey):
    for p in preds:
        if p.old_pos == prey.pos() and prey.old_pos == p.pos(): # previous was and
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
        if p.get_old_coords() == prey.get_coords() and prey.get_old_coords() == p.get_coords(): # previous was and
            return True
    return False

#functions used when the neural networks uses no comunication and is heterogeneous
def ann_inputs_outputs_T(preds, prey, net):
    preds_coords = [p.get_coords() for p in preds]
    preyx, preyy = prey.get_coords()
    input_data = []
    for x,y in preds_coords:
        #print("\npredx: ", x, "preyx:", preyx)
        #print("predy: ", y, "preyy:", preyy)
        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        #print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
        #print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
        input_data.append(norm_offsetx)
        input_data.append(norm_offsety)
    #print("INPUTS!:", input_data1)
    return net.activate(input_data)

def ann_inputs_outputs_t_T(tpreds, tprey, net):
    preds_coords = [p.position() for p in tpreds]
    preyx, preyy = tprey.position()
    input_data = []
    for x,y in preds_coords:
        #print("\npredx: ", x, "preyx:", preyx)
        #print("predy: ", y, "preyy:", preyy)
        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        #print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
        #print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
        input_data.append(norm_offsetx)
        input_data.append(norm_offsety)
    #print("INPUTS!:", input_data1)
    return net.activate(input_data)

#functions used when the neural networks uses no comunication and is homogeneous
#preds_n to garantee different order of inputs for each predator neural network
def ann_inputs_outputs_I(preds, preds_n, prey, net):
    preds_ordered = copy.deepcopy(preds)
    pred_to_order = preds_ordered[preds_n]
    del preds_ordered[preds_n]
    preds_ordered.insert(0, pred_to_order)
    
    preds_coords = [p.get_coords() for p in preds_ordered]
    preyx, preyy = prey.get_coords()
    input_data = []
    outputs= []
    for x,y in preds_coords:
        #print("\npredx: ", x, "preyx:", preyx)
        #print("predy: ", y, "preyy:", preyy)
        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        #print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
        #print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
        input_data.append(norm_offsetx)
        input_data.append(norm_offsety)
    output = net.activate(input_data)
    #print("outputs: ", outputs)
    return output
    #print("INPUTS!:", input_data1)
    #for n in range(5):
    #    outputs.append(net.activate(input_data))#random.shuffle(input_data)
    #return outputs

def ann_inputs_outputs_t_I(tpreds, tpred_n, tprey, net):
    tpreds_ordered = copy.copy(tpreds)
    tpred_to_order = tpreds_ordered[tpred_n]
    del tpreds_ordered[tpred_n]
    tpreds_ordered.insert(0, tpred_to_order)
    
    preds_coords = [p.position() for p in tpreds_ordered]
    preyx, preyy = tprey.position()
    input_data = []
    outputS = []
    for x,y in preds_coords:
        #print("\npredx: ", x, "preyx:", preyx)
        #print("predy: ", y, "preyy:", preyy)
        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        #print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
        #print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
        input_data.append(norm_offsetx)
        input_data.append(norm_offsety)
    #print("INPUTS!:", input_data1)
    return net.activate(input_data)

#new functions to handle communication in individual(homogeneous) network
def ann_inputs_outputs_signal_I(pred, signals, prey, net):
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

def ann_inputs_outputs_signal_t_I(tpred, signals, tprey, net):
    x,y = tpred.pos()
    preyx, preyy = tprey.pos()
    offsetx = toroidalDistance1coord(x, preyx, WIDTH)
    offsety = toroidalDistance1coord(y, preyy, HEIGHT)
    norm_offsetx = ((offsetx/ WIDTH) +1 )/2
    norm_offsety = ((offsety/ HEIGHT) +1 )/2
    input_data = [norm_offsetx, norm_offsety] + signals
    output, signal = tuple(net.activate(input_data))
    return output, signal

#new functions to handle communication in team(heterogeneous) neural network
def ann_inputs_outputs_signal_T(preds, signals, prey, net):
    #signal1, signal2 = signals #mudar porque o numero de sinais depende do numero de predadores e não serão sempre 2
    preds_coords = [p.get_coords() for p in preds]
    preyx, preyy = prey.get_coords()
    input_data = []
    outputs= []
    contpred= 0
    for x,y in preds_coords:
        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        input_data = input_data + [norm_offsetx, norm_offsety]
    #print("tuple(net.activate(input_data)):", tuple(net.activate(input_data)))
    input_data = input_data + signals
    outputs, signals = tuple(net.activate(input_data))
    return outputs, signals

def ann_inputs_outputs_signal_t_T(tpreds, signals, tprey, net):
    tpreds_coords = [p.pos() for p in tpreds]
    tpreyx, tpreyy = tprey.pos()
    input_data = []
    outputs= []
    contpred= 0
    for x,y in tpreds_coords:
        offsetx = toroidalDistance1coord(x, tpreyx, WIDTH)
        offsety = toroidalDistance1coord(y, tpreyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        input_data = input_data + [norm_offsetx, norm_offsety]
    #print("tuple(net.activate(input_data)):", tuple(net.activate(input_data)))
    input_data = input_data + signals
    outputs, signals = tuple(net.activate(input_data))
    return outputs, signals



#######################################################################################################################################################################################


#Unused funcs and different versions of used functions
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

def atravessaram_obsolete(preds, prey):
    for p in preds:
        if p.old_pos == prey.pos() or prey.old_pos == p.pos(): # previous was a and condition
            return True
        #following condition might be wrong used for cases where the preds move faster than prey
        #if prey.pos()[0] in range(p.old_pos[0], p.pos()[0]) or prey.pos()[1] in range(p.old_pos[1], p.pos()[1]): #doesnt work for floats
        if p.old_pos[0] < prey.pos()[0] < p.pos()[0] and p.old_pos[1] < prey.pos()[1] < p.pos()[1]:
            return True
    return False

def atravessaram_a_obsolete(preds, prey):
    for p in preds:
        if p.get_old_coords() == prey.get_coords() or prey.get_old_coords() == p.get_coords(): # previous was a and condition
            return True
        #following condition might be wrong used for cases where the preds move faster than prey
        #if prey.get_coords()[0] in range(p.get_old_coords()[0], p.get_coords()[0]) or prey.get_coords()[1] in range(p.get_old_coords()[1], p.get_coords()[1]): #doesnt work for floats
        if p.get_old_coords()[0] < prey.get_coords()[0] < p.get_coords()[0] and p.get_old_coords()[1] < prey.get_coords()[1] < p.get_coords()[1]:
            return True
    return False    


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

#def ann_inputs_outputs(preds, prey, net):
#    preds_coords = [p.get_coords() for p in preds]
#    preyx, preyy = prey.get_coords()
#    input_data = []
#    outputs= []
#    for x,y in preds_coords:
#        #print("\npredx: ", x, "preyx:", preyx)
#        #print("predy: ", y, "preyy:", preyy)
#        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
#        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
#        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
#        norm_offsety = ((offsety/ HEIGHT) +1 )/2
#        #print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
#        #print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
#        input_data.append(norm_offsetx)
#        input_data.append(norm_offsety)
#    outputs = net.activate(input_data)
#    #print("outputs: ", outputs)
#    return outputs
#    #print("INPUTS!:", input_data1)
#    #for n in range(5):
#    #    outputs.append(net.activate(input_data))#random.shuffle(input_data)
#    #return outputs
#
#def ann_inputs_outputs_t(tpreds, tprey, net):
#    preds_coords = [p.position() for p in tpreds]
#    preyx, preyy = tprey.position()
#    input_data = []
#    outputs = []
#    for x,y in preds_coords:
#        #print("\npredx: ", x, "preyx:", preyx)
#        #print("predy: ", y, "preyy:", preyy)
#        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
#        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
#        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
#        norm_offsety = ((offsety/ HEIGHT) +1 )/2
#        #print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
#        #print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
#        input_data.append(norm_offsetx)
#        input_data.append(norm_offsety)
#    #print("INPUTS!:", input_data1)
#    return net.activate(input_data)