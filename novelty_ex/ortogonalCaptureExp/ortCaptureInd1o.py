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

from collections import deque

from agents import Agent
import turtle

#importing all functions needed
from funcs import *

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
# Define a global novelty archive
NOVELTY_ARCHIVE = deque(maxlen=500)  # Set a limit to the archive size
FITNESS_THRESHOLD = 18
NOVELTY_THRESHOLD = 3
NOVELTY_THRESHOLD_TIMEOUT = 0
BEST_FITNESS_SCORE = [None ,-1, -1]

#population amount
POP_N = 500
#generations amount
MAX_N = 500
GEN_N = 0 #contador usado no eval_genome()


#simula
def simula(net, preds, preys, preys_test, height, width, ticks):
    cont = 0
    for prey in preys:
        cont+=1
        print("simulação",cont)
        simula1(net, preds, prey, height, width, ticks, cont)
    #testing best genome in new situations with new prey positions
    print("testing best genome in new situations with new prey positions")
    #for prey in preys_test:
    #    simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), height,width, ticks)

def simula1(net, preds_def, prey, height, width, ticks, cont):
    
    preds = copy.deepcopy(preds_def)#preds
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
        
        for npred in range(n_preds):#for pred, signal in zip (preds, signals):
            tpred = tpreds[npred]
            output= ann_inputs_outputs_t_I(tpreds, npred, tprey, net)[0]
            tpred_move(tpred, output, STEP)
            #to move 2 times making it move twice as fast
            #tpred_move(tpred, output, STEP)
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
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ (10*STEP))
            map.clearscreen()

            frames[0].save("gifs\\predatorTrialSuccess" + str(cont) +"_NoComInd1o.gif",
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
    frames[0].save("gifs\\best_genomeTrialRun" + str(cont) +"_NoComInd1o.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=2)  # Set loop to 0 for infinite loop


#calculo da média dos fitness das várias avaliações de cada rede neuronal numa lista de preys
def eval_fitness(net, preds_def, preys_def, height, width, ticks):
    the_fitness= 0
    behaviours = []
    #the_behaviour = (0,0)
    #cycle_count = 0
    #print([p.get_coords() for p in preys_def])
    for prey in preys_def:
        #cycle_count += 1
        #print("CYCLE:", cycle_count)
        f1, behaviour = eval_fitness1(net, preds_def, prey, height, width, ticks)
        behaviours.append(behaviour)
        #print("eval1", f1)
        the_fitness += f1
        #the_behaviour += behaviour
    #print("the summed fitness:", the_fitness, "the number of experiments per genome:", len(preys_def))
    return the_fitness/ len(preys_def), behaviours


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
    for count in range(ticks): #(500 * 2 / 10) * (3/2) = 150
        
        for npred in range(n_preds):#for pred, signal in zip (preds, signals):
            pred = preds[npred]
            output= ann_inputs_outputs_I(preds, npred, prey, net)[0]
            #no simula por aqui para mudar a cor do agente de acordo com o sinal
            pred_move(pred, output, STEP)
            #to move 2 times making it move twice as fast
            #pred_move(pred, output, STEP)

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
            print("fitness:", (2*(width + height) - mediafinaldists)/ (10*STEP))
            #the_behaviour = tuple(finaldists)
            the_behaviour = sorted([toroidalDistance_signal(prey.get_coords(), pred.get_coords(), DIST) for pred in preds])
            the_behaviour2 = [y for x in the_behaviour for y in x]
            return ((2*(width + height) - mediafinaldists)/ (10*STEP)), the_behaviour2 # max threshold is 140 ((1400 - 0) / 10)

    #new code to avoid bad genomes or neural networks that make preds only move in one direction the whole time or all preds move same direction the whole time
    #predsposx = []
    #predsposy = []
    #onedirectionpreds = []
    #for pred in preds:
    #    x,y = pred.get_coords()
    #    x_initial, y_initial = pred.initial_coords
    #    predsposx.append(x)
    #    predsposy.append(y)
    #    if x == x_initial or y == y_initial:
    #        onedirectionpreds.append(True)
    #    else:
    #        onedirectionpreds.append(False)
    #equalsx = all(i == predsposx[0] for i in predsposx)
    #equalsy = all(i == predsposy[0] for i in predsposy)
    #if all(i == True for i in onedirectionpreds):
    #    return 1 #bad fitness if all preds only moved in one direction during evolution
    #if equalsx or equalsy:
    #    return 1 #bad fitness if not capture and pos of preds in x or y are equal

    inidists = [toroidalDistance_coords(pred.get_initial_coords(), prey.get_coords(), height, width) for pred in preds]
    mediainidists = sum(inidists) / n_preds

    finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), height, width) for pred in preds]
    mediafinaldists = sum(finaldists) / n_preds
    #print("fitness:",1/(dist1 + dist2 + dist3 + dist4))
    #print("fitness:", (mediainidists - mediafinaldists) / 10)
    #the_behaviour = tuple(finaldists)
    the_behaviour = sorted([toroidalDistance_signal(prey.get_coords(), pred.get_coords(), DIST) for pred in preds])
    the_behaviour2 = [y for x in the_behaviour for y in x]
    return (mediainidists - mediafinaldists) / (10*STEP), the_behaviour2


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


###### eval genomes ####################

def eval_genomes_a(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    PREY_PREDATOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    #genome_count = 0
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT

    genome_added_n = 0
    best_generation_fitness = (0, 0)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, DIST, NOVELTY_THRESHOLD, NOVELTY_ARCHIVE, None)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        #avoid false solution trigger from novelty
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = FITNESS_THRESHOLD -1 

        # Add behavior to the archive if it is sufficiently novel
        if novelty_score >= NOVELTY_THRESHOLD:
            NOVELTY_ARCHIVE.append(the_behaviour)
            dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")
            genome_added_n += 1

        if fitness_result >= FITNESS_THRESHOLD or GEN_N == MAX_N-1:# or fitness_result > novelty_score: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif genome_count == GEN_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result)

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far: ", BEST_FITNESS_SCORE[1:])
    print("this generation best fitness result:(genome_id, fitness_result)", best_generation_fitness)
    print("genomes' behaviours added to novelty archive: ",genome_added_n)
    if genome_added_n == 0:
        NOVELTY_THRESHOLD_TIMEOUT += 1
    else:
        NOVELTY_THRESHOLD_TIMEOUT = 0

    if NOVELTY_THRESHOLD_TIMEOUT >= 2:
        NOVELTY_THRESHOLD *= 0.9
        print("NOVELTY THRESHOLD REDUCED")
    if genome_added_n >=4:
        NOVELTY_THRESHOLD *= 1.2
        print("NOVELTY THRESHOLD INCREASED")
    print("NOVELTY_THRESHOLD: ", NOVELTY_THRESHOLD, "\n")    
    GEN_N +=1


def eval_genomes_ag(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    PREY_PREDATOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    #genome_count = 0
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT
    gen_behaviours = deque(maxlen=500)
    genome_added_n = 0
    best_generation_fitness = (0, 0)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, DIST, NOVELTY_THRESHOLD, NOVELTY_ARCHIVE, gen_behaviours)
        gen_behaviours.append(the_behaviour)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        #avoid false solution trigger from novelty
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = FITNESS_THRESHOLD -1

        # Add behavior to the archive if it is sufficiently novel
        if novelty_score >= NOVELTY_THRESHOLD:
            NOVELTY_ARCHIVE.append(the_behaviour)
            dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")
            genome_added_n += 1


        if fitness_result >= FITNESS_THRESHOLD or GEN_N == MAX_N-1:# or fitness_result > novelty_score: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif genome_count == GEN_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result)

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far: ", BEST_FITNESS_SCORE[1:])
    print("this generation best fitness result:(genome_id, fitness_result)", best_generation_fitness)
    print("genomes' behaviours added to novelty archive: ",genome_added_n)
    if genome_added_n == 0:
        NOVELTY_THRESHOLD_TIMEOUT += 1
    else:
        NOVELTY_THRESHOLD_TIMEOUT = 0

    if NOVELTY_THRESHOLD_TIMEOUT >= 2:
        NOVELTY_THRESHOLD *= 0.9
        print("NOVELTY THRESHOLD REDUCED")
    if genome_added_n >=4:
        NOVELTY_THRESHOLD *= 1.2
        print("NOVELTY THRESHOLD INCREASED")
    print("NOVELTY_THRESHOLD: ", NOVELTY_THRESHOLD, "\n")
    
    GEN_N +=1


def eval_genomes_gen(genomes, config):
    #genome_count = 0
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD_TIMEOUT

    gen_behaviours = []
    best_generation_fitness = (0, 0)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, DIST, NOVELTY_THRESHOLD, None, gen_behaviours)
        gen_behaviours.append(the_behaviour)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        #avoid false solution trigger from novelty
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = FITNESS_THRESHOLD -1

        if fitness_result >= FITNESS_THRESHOLD or GEN_N == MAX_N-1:# or fitness_result > novelty_score: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif genome_count == GEN_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result)
    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far: ", BEST_FITNESS_SCORE[1:])
    print("this generation best fitness result:(genome_id, fitness_result)", best_generation_fitness)
    
    GEN_N +=1

def eval_genomes_r(genomes, config):
     #genome_count = 0
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD

    best_generation_fitness = (0, 0)
    gen_behaviours = []
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, DIST, NOVELTY_THRESHOLD, NOVELTY_ARCHIVE, None)
        gen_behaviours.append(the_behaviour)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        #avoid false solution trigger from novelty
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = FITNESS_THRESHOLD -1 

        if fitness_result >= FITNESS_THRESHOLD or GEN_N == MAX_N-1:# or fitness_result > novelty_score: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif genome_count == GEN_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result)

    #each generation pick one random behaviour from genomes
    NOVELTY_ARCHIVE.append(random.choice(gen_behaviours))
    dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far: ", BEST_FITNESS_SCORE[1:])
    print("this generation best fitness result:(genome_id, fitness_result)", best_generation_fitness)
    GEN_N +=1

def eval_genomes_checkpoint_a(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    PREY_PREDATOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    #genome_count = 0
    NOVELTY_ARCHIVE = load("out\\novelty_archive.pkl")
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT

    genome_added_n = 0
    best_generation_fitness = (0, 0)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, DIST, NOVELTY_THRESHOLD, NOVELTY_ARCHIVE, None)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        #avoid false solution trigger from novelty
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = FITNESS_THRESHOLD -1 

        # Add behavior to the archive if it is sufficiently novel
        if novelty_score >= NOVELTY_THRESHOLD:
            NOVELTY_ARCHIVE.append(the_behaviour)
            dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")
            genome_added_n += 1

        if fitness_result >= FITNESS_THRESHOLD or GEN_N == MAX_N-1:# or fitness_result > novelty_score: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif genome_count == GEN_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result)

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far: ", BEST_FITNESS_SCORE[1:])
    print("this generation best fitness result:(genome_id, fitness_result)", best_generation_fitness)
    print("genomes' behaviours added to novelty archive: ",genome_added_n)
    if genome_added_n == 0:
        NOVELTY_THRESHOLD_TIMEOUT += 1
    else:
        NOVELTY_THRESHOLD_TIMEOUT = 0

    if NOVELTY_THRESHOLD_TIMEOUT >= 2:
        NOVELTY_THRESHOLD *= 0.9
        print("NOVELTY THRESHOLD REDUCED")
    if genome_added_n >=4:
        NOVELTY_THRESHOLD *= 1.2
        print("NOVELTY THRESHOLD INCREASED")
    print("NOVELTY_THRESHOLD: ", NOVELTY_THRESHOLD, "\n")
    
    GEN_N +=1

def eval_genomes_checkpoint_ag(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list. 
    The provided configuration is used to create feed-forward 
    neural network from each genome and after that created
    the neural network evaluated in its ability to solve
    PREY_PREDATOR problem. As a result of this function execution, the
    the fitness score of each genome updated to the newly
    evaluated one.
    Arguments:
        genomes: The list of genomes from population in the 
                current generation
        config: The configuration settings with algorithm
                hyper-parameters
    """
    #genome_count = 0
    NOVELTY_ARCHIVE = load("out\\novelty_archive.pkl")
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT

    gen_behaviours = deque(maxlen=500)
    genome_added_n = 0
    best_generation_fitness = (0, 0)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, DIST, NOVELTY_THRESHOLD, NOVELTY_ARCHIVE, gen_behaviours)
        gen_behaviours.append(the_behaviour)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        #avoid false solution trigger from novelty
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = FITNESS_THRESHOLD -1

        # Add behavior to the archive if it is sufficiently novel
        if novelty_score >= NOVELTY_THRESHOLD:
            NOVELTY_ARCHIVE.append(the_behaviour)
            dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")
            genome_added_n += 1


        if fitness_result >= FITNESS_THRESHOLD or GEN_N == MAX_N-1:# or fitness_result > novelty_score: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif genome_count == GEN_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result)
            
    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far: ", BEST_FITNESS_SCORE[1:])
    print("this generation best fitness result:(genome_id, fitness_result)", best_generation_fitness)
    print("genomes' behaviours added to novelty archive: ",genome_added_n)
    if genome_added_n == 0:
        NOVELTY_THRESHOLD_TIMEOUT += 1
    else:
        NOVELTY_THRESHOLD_TIMEOUT = 0

    if NOVELTY_THRESHOLD_TIMEOUT >= 2:
        NOVELTY_THRESHOLD *= 0.9
        print("NOVELTY THRESHOLD REDUCED")
    if genome_added_n >=4:
        NOVELTY_THRESHOLD *= 1.2
        print("NOVELTY THRESHOLD INCREASED")
    print("NOVELTY_THRESHOLD: ", NOVELTY_THRESHOLD, "\n")
    
    GEN_N +=1

### RUNNING ##########################################

def run_experiment(config_file, genomeloadfile = None):
    """
    The function to run XOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
    global BEST_FITNESS_SCORE
    dump(PREYS_9, "out\\savedPREYS.pkl")
    loaded_genome_s = None
    #part where it is possible to load previously trained genomes
    if genomeloadfile != None:
        # Later, you can load the genome from the file
        with open(genomeloadfile, 'rb') as f:
                #loaded_genome = neat.Genome()
                #loaded_genome.parse(f)
            loaded_genome_s = pickle.load(f)

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    #loading trained genomes into population
    if isinstance(loaded_genome_s, list):
        for gen, gen_n in zip(loaded_genome_s, range(500)):
            p.population[gen_n] = gen
    elif loaded_genome_s != None:
        p.population[0] = loaded_genome_s

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(out_dir, 'neat-checkpoint-')))


    # Run for up to 300 generations.
    best_genome = p.run(eval_genomes_a, MAX_N)#500
    the_best_genome = BEST_FITNESS_SCORE[0]

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(the_best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'offx2', -4: 'offy2', -5: 'offx3', -6: 'offy3', 0:'Move_outputp'}
    visualize.draw_net(config, the_best_genome, True, node_names=node_names, directory=out_dir)
    #print("AQUI!!!")
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))

    #print("ALI!!!")
    #save image of plot of the fitness
    #plot1 = gw.getWindowsWithTitle("Figure 1")[0]
    #com,larg =plot1.size
    #plot1.moveTo(10,10)
    #image1 = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
    #image1.save("results\ortogonalPredatorsProblemResultsFitness.png")
    #print("Acoli!!!")

    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

    #save image of plot of the species
    #plot2 = gw.getWindowsWithTitle("Figure 1")[0]
    #com2,larg2 =plot2.size
    #plot2.moveTo(500, 500)
    #image2 = ImageGrab.grab(bbox=(500, 500, 500+com2, 500+larg2))
    #image2.save("results\ortogonalPredatorsProblemResultsSpecies.png")

    return the_best_genome


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)

#MUDAR COM URGÊNCIA PORQUE VAI BUSCAR GENOMAS AO FICHEIRO QUE não TÊM nada a ver COM o MELHOR GENOMA ATUAL
def nrunexperiment(n, genomeloadfile = None):
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

            #run as normal or considering previously trained genomes
            if genomeloadfile == None:
                # Run the experiment
                best_genome = run_experiment(config_path)
                #the_best_genome = BEST_FITNESS_SCORE[0]
            else: 
                best_genome = run_experiment(config_path, genomeloadfile)
                #the_best_genome = BEST_FITNESS_SCORE[0]
            print("best_genome.fitness:", best_genome.fitness)
            #to get the best genome out of the n resulting best genomes of the n experiments
            if best_genome.fitness > best_of_the_bestGenomefitness:
                best_of_the_bestGenomefitness = best_genome.fitness
                best_of_the_bestGenome = best_genome
                
            # Assuming 'genome' is your NEAT genome object
            genome_path = 'storedgenomes\\goodgenomes_NoComInd1o.pkl'

            # Save the genome to a file
            with open("storedgenomes\\goodgenomes_NoComInd1o.pkl", "wb") as f:
                pickle.dump(best_genome, f)
                f.close()

            if i < n-1:
                userinput = str(input("want to carry on with the program?:(y/n) "))
                if userinput == "n":
                    break
    #keep the best genome of the n experimentations in a separate file
    best_genome_path = 'storedgenomes\\bestgenome_NoComInd1o.pkl'
    with open("storedgenomes\\bestgenome_NoComInd1o.pkl", "wb") as f:
            pickle.dump(best_of_the_bestGenome, f)
            f.close()
    print("best_of_the_bestGenome.fitness", best_of_the_bestGenome.fitness)
    print("end of regular experimentation!")
    print()

    print("simulate behavior of best genome on the trained set:", best_of_the_bestGenome)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    net = neat.nn.FeedForwardNetwork.create(best_of_the_bestGenome, config)
    simula(net, PREDS_DEF, PREYS_9, PREYS_DEF, HEIGHT, WIDTH, TICKS)

    print("simulate behavior of best genome on a new set of randomly placed preys to test the model")
    simula(net, PREDS_DEF, PREYS_TEST, PREYS_DEF, HEIGHT, WIDTH, TICKS)
    print("The END.")


#loading checkpoint for continuation
def runCheckpointExperiment(filename, check_n):

    global GEN_N
    global MAX_N
    global BEST_FITNESS_SCORE
    PREYS_9= load("out\\savedPREYS.pkl")
    BEST_FITNESS_SCORE = load("out\\BEST_FITNESS_GENOME.pkl")
    GEN_N = check_n
    gen_to_run  = MAX_N - GEN_N 
    config_path = os.path.join(local_dir, 'exercise.ini')

    restoredPopulation = neat.Checkpointer.restore_checkpoint(filename)

    # Add a stdout reporter to show progress in the terminal.
    restoredPopulation.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    restoredPopulation.add_reporter(stats)
    restoredPopulation.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(out_dir, 'neat-checkpoint-')))

    best_genome = restoredPopulation.run(eval_genomes_checkpoint_a, gen_to_run)
    the_best_genome = BEST_FITNESS_SCORE[0]

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(the_best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'offx2', -4: 'offy2', -5: 'offx3', -6: 'offy3', 0:'Move_outputp'}
    visualize.draw_net(restoredPopulation.config, the_best_genome, True, node_names=node_names, directory=out_dir)

    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))

    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

    #keep the best genome of the n experimentations in a separate file
    best_genome_path = 'storedgenomes\\bestgenome_NoComTeam1o.pkl'
    with open("storedgenomes\\bestgenome_NoComTeam1o.pkl", "wb") as f:
            pickle.dump(the_best_genome, f)
            f.close()

    print("best_of_the_bestGenome.fitness", the_best_genome.fitness)
    print("end of regular experimentation!")
    print()

    print("simulate behavior of best genome on the trained set:", the_best_genome)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    net = neat.nn.FeedForwardNetwork.create(the_best_genome, config)
    simula(net, PREDS_DEF, PREYS_9, PREYS_DEF, HEIGHT, WIDTH, TICKS)

### RUNNING END #################################################


nrunexperiment(1)
#nrunexperiment(1, "storedgenomes\\goodgenomes_NoComInd1o.pkl")

#checkpointfile = "out\\neat-checkpoint-498"
#runCheckpointExperiment(checkpointfile, 498)