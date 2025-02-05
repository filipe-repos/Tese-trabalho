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

from collections import deque

#importing all functions needed
from funcs import *

TICKS = int(((HEIGHT*2) / STEP) * 1.5)
N_PREDS = 4

local_dir = os.path.dirname(__file__)
#the list of Preds to be used in in each avaliation
PREDS_DEF = createpredators_bottom(HEIGHT, WIDTH, N_PREDS, STEP)
PREYS_9= load("out\\savedPREYS.pkl")
BEST_FITNESS_SCORE = load("out\\BEST_FITNESS_GENOME.pkl")
config_path = os.path.join(local_dir, 'exercise.ini')

the_best_genome = BEST_FITNESS_SCORE[0]
print("best genome data:\n", BEST_FITNESS_SCORE[1:])



#simula
def simula(net, preds, preys, height, width, ticks):
    cont = 0
    for prey in preys:
        cont+=1
        print("simulação",cont)
        simula1_t(net,copy.deepcopy(preds), copy.deepcopy(prey), height, width, ticks, cont)
    #testing best genome in new situations with new prey positions
    print("testing best genome in new situations with new prey positions")

def simula1_i(net, preds, prey, height, width, ticks, cont):
    
    preds = copy.deepcopy(preds)#preds
    prey = copy.deepcopy(prey)
    prey_coords = prey.get_coords()

    colors = ["yellow", "orange", "red", "black", "yellow", "orange", "red", "black"]
    n_preds = len(preds)
    

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
        newagentpos = []
        for npred in range(n_preds):#for pred, signal in zip (preds, signals):
            tpred = tpreds[npred]
            output= ann_inputs_outputs_t_I(tpreds, npred, tprey, net)[0]

            newagentpos.append((tpred,tpred_newpos(tpred, output, STEP)))
            
            #to move 2 times making it move twice as fast
            #tpred_move(tpred, output, STEP)
            #print("tpred pos:", tpred.pos())

        #print("pred new coords: ", tpred.position())
        #To make the prey not move commented the method function to make it move
        newagentpos.append((tprey,tprey_newpos(tprey,tpreds, STEP)))

        #to delete just to test something
        #print("prey pos:", tprey.pos())

        newagentmovements = limpamovimentos_t(newagentpos)
        #só se movem os agentes que não vão para a mesma casa nem que se atravessam
        for agent, agentnewpos in newagentmovements:
            agent_go(agent, agentnewpos, STEP)

        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)

        if captura_t(tpreds, tprey):
            print("presa apanhada!!!")

            finaldists = [toroidalDistance_coords(tpred.position(), tprey.position(), HEIGHT, WIDTH) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds

            #print("fitness:", 1)
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ (10*STEP))
            map.clearscreen()

            frames[0].save("out\\gifs\\predatorTrialSuccess" + str(cont) +"_NoComInd1o.gif",
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
    print("fitness:", (mediainidists - mediafinaldists) / (10*STEP))
    frames[0].save("out\\gifs\\best_genomeTrialRun" + str(cont) +"_NoComInd1o.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=2)  # Set loop to 0 for infinite loop


def simula1_t(net, preds, prey, height, width, ticks, cont):
    
    preds = copy.deepcopy(preds)#preds
    prey = copy.deepcopy(prey)
    prey_coords = prey.get_coords()

    colors = ["yellow", "orange", "red", "black", "yellow", "orange", "red", "black"]
    n_preds = len(preds)

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
        newagentpos = []
        outputs = ann_inputs_outputs_t_T(tpreds, tprey, net)
        for tpred, output in zip(tpreds, outputs):
            newagentpos.append((tpred,tpred_newpos(tpred, output, STEP)))
            #to move 2 times making it move twice as fast
            #tpred_move(tpred, output, STEP)

        #print("pred new coords: ", tpred.position())
        #To make the prey not move commented the method function to make it move
        newagentpos.append((tprey,tprey_newpos(tprey,tpreds, STEP)))
        #new_tprey_move(tprey, tpreds, STEP)

        newagentmovements = limpamovimentos_t(newagentpos)
        #só se movem os agentes que não vão para a mesma casa nem que se atravessam
        for agent, agentnewpos in newagentmovements:
            agent_go(agent, agentnewpos, STEP)

        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)

        if captura_t(tpreds, tprey):
            print("presa apanhada!!!")

            finaldists = [toroidalDistance_coords(tpred.position(), tprey.position(), HEIGHT, WIDTH) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds

            #print("fitness:", 1)
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ (10*STEP))
            map.clearscreen()

            frames[0].save("out\\gifs\\predatorTrialSuccess" + str(cont) +"_NoComTeam1o.gif",
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
    print("fitness:", (mediainidists - mediafinaldists) / (10*STEP))
    frames[0].save("out\\gifs\\best_genomeTrialRun" + str(cont) +"_NoComTeam1o.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=2)  # Set loop to 0 for infinite loop


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

net = neat.nn.FeedForwardNetwork.create(the_best_genome, config)

print("simulate behavior of best genome on the trained set:", the_best_genome)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)
net = neat.nn.FeedForwardNetwork.create(the_best_genome, config)
simula(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)