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

import pickle

from agents import Agent
import turtle

from collections import deque

#importing all functions needed
from funcs import *

#DIST is The dimensions of map, onlyapplicable if map is square
DIST = 350
#width of map
WIDTH = 350
#height of map
HEIGHT = 350
#Step how much the agents(preds and prey) move per turn. Should allways be DIST/50
STEP = DIST/25
#N_Evals is the number of different evaluations(cases where prey is in different position) done per genome per generation
N_EVALS = 9
#N_PREDS is the number of predators to be present in experiment and to chase the prey
N_PREDS = 3
#TICKS is the limit of turns allowed or movements to be done by all agents before the experiment ends
#TICKS = int(((HEIGHT*2) / STEP) * 1.5)
TICKS = 150
# Define a global novelty archive
NOVELTY_ARCHIVE = deque(maxlen=1500)  # Set a limit to the archive size
FITNESS_THRESHOLD = 1000000
NOVELTY_THRESHOLD = 4
NOVELTY_THRESHOLD_TIMEOUT = 0
BEST_FITNESS_SCORE = [None ,-1, -1, -1, -1]

#population amount
POP_N = 500
#generations amount
MAX_N = 1500
GEN_N = 0 #contador usado no eval_genome()


#calculo da média dos fitness das várias avaliações de cada rede neuronal numa lista de preys
def eval_fitness(net, preds_def, preys_def, height, width, ticks):
    the_fitness= 0
    behaviours = []
    capturas = []
    #the_behaviour = (0,0)
    #cycle_count = 0
    #print([p.get_coords() for p in preys_def])
    for prey in preys_def:
        #cycle_count += 1
        #print("CYCLE:", cycle_count)
        f1, behaviour, captura = eval_fitness1(net, preds_def, prey, height, width, ticks)
        behaviours.append(behaviour)
        capturas.append(captura)
        #print("eval1", f1)
        the_fitness += f1
        #the_behaviour += behaviour
    #print("the summed fitness:", the_fitness, "the number of experiments per genome:", len(preys_def))
    return the_fitness/ len(preys_def), behaviours, capturas


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
    the_behaviour = []
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
            output, signal = ann_inputs_outputs_signal_I(pred, sinaisOutrem, prey, net)
            signals[npredsig] = signal
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
            print("fitness:", (2*(width + height) - mediafinaldists)/ (5*STEP))
            print()
            #the_behaviour = tuple(finaldists)
            behaviour = sorted([toroidalDistance_signal(prey.get_coords(), pred.get_coords(), DIST) for pred in preds])
            behaviour2 = [y for x in behaviour for y in x]
            the_behaviour.append(behaviour2)

            if count <= int(ticks/2):
                n_behaviour = [ (0.0, 0.0) for pred in preds]
                n_behaviour2 = [y for x in n_behaviour for y in x]
                the_behaviour.append(n_behaviour2)
            #the_behaviour.append([pred.get_coords() for pred in preds])
            return ((2*(width + height) - mediafinaldists)/ (5*STEP)), the_behaviour, True # max threshold is 160 ((1600 - 0) / 10)

        #add instance of behaviour when experience is at half
        if count == int(ticks/2):
            l_behaviour = sorted([toroidalDistance_signal(prey.get_coords(), pred.get_coords(), DIST) for pred in preds])
            l_behaviour2 = [y for x in l_behaviour for y in x]
            the_behaviour.append(l_behaviour2)
    

    inidists = [toroidalDistance_coords(pred.get_initial_coords(), prey.get_coords(), height, width) for pred in preds]
    mediainidists = sum(inidists) / n_preds

    finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), height, width) for pred in preds]
    mediafinaldists = sum(finaldists) / n_preds
    #print("fitness:",1/(dist1 + dist2 + dist3 + dist4))
    #print("fitness:", (mediainidists - mediafinaldists) / 10)
    #the_behaviour = tuple(finaldists)
    t_behaviour = sorted([toroidalDistance_signal(prey.get_coords(), pred.get_coords(), DIST) for pred in preds])
    t_behaviour2 = [y for x in t_behaviour for y in x]
    the_behaviour.append(t_behaviour2)
    #print("the_real_behaviour", the_behaviour)
    return (mediainidists - mediafinaldists) / (5*STEP), the_behaviour, False


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

if N_EVALS != 9:
    PREYS_9 = PREYS_DEF

###### eval genomes ####################

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
    global NOVELTY_ARCHIVE

    genome_added_n = 0
    best_generation_fitness = (0, 0)
    gen_behaviours = deque(maxlen=500)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #além do fitness result tenho de ter uma lista de nove boleanos a indicar se houve captura por ensaio
        # e se houve não calculo o novelty desse individuo mas ponho um valor colossal (>= FITNESS_THRESHOLD) para sair com solução 
        fitness_result, the_behaviour, capturas = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        #this_generation_behaviours[genome_id] = the_behaviour
        #gen_behaviours.append((genome_id, genome, the_behaviour, capturas))
        gen_behaviours.append(the_behaviour)
        
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result, GEN_N, capturas, the_behaviour)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result, GEN_N, capturas, the_behaviour)

        if all(capturas):
            genome.fitness = FITNESS_THRESHOLD+1
        #Tenho de associar capturas e behaviours ao genome_id
    for genome, behaviour in zip(genomes, gen_behaviours):
        novelty_score = calculate_novelty(behaviour, DIST, 10, gen_behaviours, NOVELTY_ARCHIVE)#em vez de none todos os comportamentos desta geração inclusive ele proprio(k+1)

        # Add behavior to the archive if it is sufficiently novel
        if novelty_score >= NOVELTY_THRESHOLD:
            NOVELTY_ARCHIVE.append(the_behaviour)
            dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")
            genome_added_n += 1

        genome[1].fitness = novelty_score

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:\n) ", BEST_FITNESS_SCORE[1:], "\n")
    print("this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:\n)", best_generation_fitness, "\n")
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

    best_generation_fitness = (0, 0)
    gen_behaviours = deque(maxlen=500)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #além do fitness result tenho de ter uma lista de nove boleanos a indicar se houve captura por ensaio
        # e se houve não calculo o novelty desse individuo mas ponho um valor colossal (>= FITNESS_THRESHOLD) para sair com solução 
        fitness_result, the_behaviour, capturas = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        #this_generation_behaviours[genome_id] = the_behaviour
        #gen_behaviours.append((genome_id, genome, the_behaviour, capturas))
        gen_behaviours.append(the_behaviour)
        
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result, GEN_N, capturas, the_behaviour)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result, GEN_N, capturas, the_behaviour)

        if all(capturas):
            genome.fitness = FITNESS_THRESHOLD+1
        #Tenho de associar capturas e behaviours ao genome_id
    for genome, behaviour in zip(genomes, gen_behaviours):
        novelty_score = calculate_novelty(behaviour, DIST, 5, gen_behaviours, None)#em vez de none todos os comportamentos desta geração inclusive ele proprio(k+1)
        genome[1].fitness = novelty_score

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:\n) ", BEST_FITNESS_SCORE[1:], "\n")
    print("this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:\n)", best_generation_fitness, "\n")
    GEN_N +=1

#Mudar
def eval_genomes_r(genomes, config):
    #genome_count = 0
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT
    global NOVELTY_ARCHIVE

    best_generation_fitness = (0, 0)
    gen_behaviours = deque(maxlen=500)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #além do fitness result tenho de ter uma lista de nove boleanos a indicar se houve captura por ensaio
        # e se houve não calculo o novelty desse individuo mas ponho um valor colossal (>= FITNESS_THRESHOLD) para sair com solução 
        fitness_result, the_behaviour, capturas = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        #this_generation_behaviours[genome_id] = the_behaviour
        #gen_behaviours.append((genome_id, genome, the_behaviour, capturas))
        gen_behaviours.append(the_behaviour)
        
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result, GEN_N, capturas, the_behaviour)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result, GEN_N, capturas, the_behaviour)

        if all(capturas):
            genome.fitness = FITNESS_THRESHOLD+1
        #Tenho de associar capturas e behaviours ao genome_id
    for genome, behaviour in zip(genomes, gen_behaviours):
        novelty_score = calculate_novelty(behaviour, DIST, 10, gen_behaviours, NOVELTY_ARCHIVE)#em vez de none todos os comportamentos desta geração inclusive ele proprio(k+1)
        genome[1].fitness = novelty_score

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:\n) ", BEST_FITNESS_SCORE[1:], "\n")
    print("this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:\n)", best_generation_fitness, "\n")

    f = open("out\\gen_evo_info.txt", "a")
    f.write("GEN " + str(GEN_N)  + "\n" + "best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:)\n " +  str(BEST_FITNESS_SCORE[1:]) + "\n" + "this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:)\n" + str(best_generation_fitness) + "\n")
    f.close()

    #each generation pick one random behaviour from genomes
    NOVELTY_ARCHIVE.append(random.choice(gen_behaviours))
    dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")

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
    global NOVELTY_ARCHIVE
    NOVELTY_ARCHIVE = load("out\\novelty_archive.pkl")
    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT

    genome_added_n = 0
    best_generation_fitness = (0, 0)
    gen_behaviours = deque(maxlen=500)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #além do fitness result tenho de ter uma lista de nove boleanos a indicar se houve captura por ensaio
        # e se houve não calculo o novelty desse individuo mas ponho um valor colossal (>= FITNESS_THRESHOLD) para sair com solução 
        fitness_result, the_behaviour, capturas = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        #this_generation_behaviours[genome_id] = the_behaviour
        #gen_behaviours.append((genome_id, genome, the_behaviour, capturas))
        gen_behaviours.append(the_behaviour)
        
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result, GEN_N, capturas, the_behaviour)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result, GEN_N, capturas, the_behaviour)

        if all(capturas):
            genome.fitness = FITNESS_THRESHOLD+1
        #Tenho de associar capturas e behaviours ao genome_id
    for genome, behaviour in zip(genomes, gen_behaviours):
        novelty_score = calculate_novelty(behaviour, DIST, 10, gen_behaviours, NOVELTY_ARCHIVE)#em vez de none todos os comportamentos desta geração inclusive ele proprio(k+1)

        # Add behavior to the archive if it is sufficiently novel
        if novelty_score >= NOVELTY_THRESHOLD:
            NOVELTY_ARCHIVE.append(the_behaviour)
            dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")
            genome_added_n += 1

        genome[1].fitness = novelty_score

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:\n) ", BEST_FITNESS_SCORE[1:], "\n")
    print("this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:\n)", best_generation_fitness, "\n")
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

#Mudar
def eval_genomes_checkpoint_r(genomes, config):
    #genome_count = 0
    global NOVELTY_ARCHIVE
    NOVELTY_ARCHIVE = load("out\\novelty_archive.pkl")

    global GEN_N
    print("GEN_N", GEN_N, "\n")
    global BEST_FITNESS_SCORE
    global NOVELTY_THRESHOLD  # Adjust this threshold as needed
    global NOVELTY_THRESHOLD_TIMEOUT

    best_generation_fitness = (0, 0)
    gen_behaviours = deque(maxlen=500)
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        #além do fitness result tenho de ter uma lista de nove boleanos a indicar se houve captura por ensaio
        # e se houve não calculo o novelty desse individuo mas ponho um valor colossal (>= FITNESS_THRESHOLD) para sair com solução 
        fitness_result, the_behaviour, capturas = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        #this_generation_behaviours[genome_id] = the_behaviour
        #gen_behaviours.append((genome_id, genome, the_behaviour, capturas))
        gen_behaviours.append(the_behaviour)
        
        #keep record of best fitness result in current generation
        if fitness_result > best_generation_fitness[1]:
            best_generation_fitness = (genome_id, fitness_result, GEN_N, capturas, the_behaviour)
        if fitness_result > BEST_FITNESS_SCORE[2]:
            BEST_FITNESS_SCORE = (genome ,genome_id, fitness_result, GEN_N, capturas, the_behaviour)

        if all(capturas):
            genome.fitness = FITNESS_THRESHOLD+1
        #Tenho de associar capturas e behaviours ao genome_id
    for genome, behaviour in zip(genomes, gen_behaviours):
        novelty_score = calculate_novelty(behaviour, DIST, 10, gen_behaviours, NOVELTY_ARCHIVE)#em vez de none todos os comportamentos desta geração inclusive ele proprio(k+1)
        genome[1].fitness = novelty_score

    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")
    print("best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:\n) ", BEST_FITNESS_SCORE[1:], "\n")
    print("this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:\n)", best_generation_fitness, "\n")

    f = open("out\\gen_evo_info.txt", "a")
    f.write("GEN " + str(GEN_N)  + "\n" + "best genome so far:(genome_id, fitness_result, gen, capturas, behaviour:)\n " +  str(BEST_FITNESS_SCORE[1:]) + "\n" + "this generation best fitness result:(genome_id, fitness_result, gen, capturas, behaviour:)\n" + str(best_generation_fitness) + "\n \n")
    f.close()

    #each generation pick one random behaviour from genomes
    NOVELTY_ARCHIVE.append(random.choice(gen_behaviours))
    dump(NOVELTY_ARCHIVE, "out\\novelty_archive.pkl")

    GEN_N +=1

### RUNNING ##########################################

def run_experiment(config_file, genomeloadfile = None):
    """
    The function to run PREY_PREDATOR experiment against hyper-parameters 
    defined in the provided configuration file.
    The winner genome will be rendered as a graph as well as the
    important statistics of neuroevolution process execution.
    Arguments:
        config_file: the path to the file with experiment 
                    configuration
    """
    global BEST_FITNESS_SCORE
    
    global PREYS_9
    #PREYS_9= load("out\\savedPREYS.pkl")
    dump(PREYS_9, "out\\savedPREYS.pkl")

    os.mkdir(".\out\gifs")

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
        for gen, gen_n in zip(loaded_genome_s, range(POP_N)):
            p.population[gen_n] = gen
    elif loaded_genome_s != None:
        p.population[0] = loaded_genome_s

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(out_dir, 'neat-checkpoint-')))

    # Run for up to 300 generations.
    best_genome = p.run(eval_genomes_r, MAX_N)#500
    #di_best=p.reporters.reporters[0].best_genome()
    the_fitness_genome = BEST_FITNESS_SCORE[0]

    if the_fitness_genome.fitness >= best_genome.fitness or best_genome.fitness < 10:
        the_best_genome = the_fitness_genome
    else:
        the_best_genome = best_genome

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(the_best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'signal1', -4: 'signal2', 0:'Move_outputp', 1:'Signal'}
    visualize.draw_net(config, the_best_genome, False, node_names=node_names, directory=out_dir)
    #print("AQUI!!!")
    visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.svg'))

    visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.svg'))

    return the_best_genome


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)

def nrunexperiment(n, genomeloadfile = None):
    #best_of_the_bestGenome = None
    #best_of_the_bestGenomefitness = 0
    #run this n times
    global GEN_N
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

            print("The END.")

            old_name_o = ".\out"
            new_name_o = ".\out" + str(i+1)
            os.rename(old_name_o, new_name_o)
            os.mkdir(old_name_o)

            name_g = ".\out\gifs"
            os.mkdir(name_g)
            GEN_N = 0
            global BEST_FITNESS_SCORE
            BEST_FITNESS_SCORE = [None ,-1, -1, -1, -1]


#loading checkpoint for continuation
def runCheckpointExperiment(filename, check_n):

    global GEN_N
    global MAX_N
    global PREYS_9
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

    best_genome = restoredPopulation.run(eval_genomes_checkpoint_r, gen_to_run)
    the_fitness_genome = BEST_FITNESS_SCORE[0]
    dump(BEST_FITNESS_SCORE, "out\\BEST_FITNESS_GENOME.pkl")

    if the_fitness_genome.fitness >= best_genome.fitness or best_genome.fitness < 10:
        the_best_genome = the_fitness_genome
    else:
        the_best_genome = best_genome

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(the_best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'signal1', -4: 'signal2', 0:'Move_outputp', 1:'Signal'}
    visualize.draw_net(restoredPopulation.config, the_best_genome, False, node_names=node_names, directory=out_dir)

    visualize.plot_stats(stats, ylog=False, view=False, filename=os.path.join(out_dir, 'avg_fitness.svg'))

    visualize.plot_species(stats, view=False, filename=os.path.join(out_dir, 'speciation.svg'))

    #keep the best genome of the n experimentations in a separate file
    #best_genome_path = 'storedgenomes\\bestgenome_NoComTeam1o.pkl'
    #with open("storedgenomes\\bestgenome_NoComTeam1o.pkl", "wb") as f:
    #        pickle.dump(the_best_genome, f)
    #        f.close()
    #
    print("best_of_the_bestGenome.fitness", the_best_genome.fitness)
    print("end of regular experimentation!")
    print()


### RUNNING END #################################################

nrunexperiment(9)
#nrunexperiment(1, "storedgenomes\\goodgenomes_SignalInd.pkl")

#checkpointfile = "out\\neat-checkpoint-744"
#runCheckpointExperiment(checkpointfile, 744)
