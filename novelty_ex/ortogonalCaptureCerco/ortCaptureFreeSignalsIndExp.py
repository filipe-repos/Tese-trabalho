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
STEP = 7
#N_Evals is the number of different evaluations(cases where prey is in different position) done per genome per generation
N_EVALS = 1
#N_PREDS is the number of predators to be present in experiment and to chase the prey
N_PREDS = 4
#TICKS is the limit of turns allowed or movements to be done by all agents before the experiment ends
TICKS = int(((HEIGHT*2) / STEP) * 1.5)
# Define a global novelty archive
NOVELTY_ARCHIVE = deque(maxlen=5000)  # Set a limit to the archive size
FITNESS_THRESHOLD = 18

#population amount
POP_N = 500
#generations amount
MAX_N = 499
GEN_N = 0 #contador usado no eval_genome()


#simula
def simula(net, preds, preys, preys_test, height, width, ticks):
    cont = 0
    for prey in preys:
        cont+=1
        print("simulação",cont)
        simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), height, width, ticks, cont)
    #testing best genome in new situations with new prey positions
    print("testing best genome in new situations with new prey positions")
    #for prey in preys_test:
    #    simula1(net,copy.deepcopy(preds), copy.deepcopy(prey), height,width, ticks)

def simula1(net, preds, prey, height, width, ticks, cont):
    
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
        newagentpos = []
        for npredsig in range(n_preds):#for pred, signal in zip (preds, signals):
            tpred = tpreds[npredsig]
            sinaisOutrem =  copy.deepcopy(signals)
            del sinaisOutrem[npredsig]
            output, signal = ann_inputs_outputs_signal_t_I(tpred, sinaisOutrem, tprey, net)
            signals[npredsig] = signal

            newagentpos.append((tpred,tpred_newpos(tpred, output, STEP)))
            #print("tpred pos:", tpred.pos())

        #print("pred new coords: ", tpred.position())
        
        #To make the prey move, the method function to make it move
        newagentpos.append((tprey,tprey_newpos(tprey,tpreds, STEP)))

        #To make the prey not move use this line bellow instead of 102, the method function to make it move
        #newagentpos.append((tprey, tprey.pos()))

        newagentmovements = limpamovimentos_t(newagentpos)

        #só se movem os agentes que não vão para a mesma casa nem que se atravessam
        for agent, agentnewpos in newagentmovements:
            agent_go(agent, agentnewpos, STEP)
        
        #to delete just to test something
        #print("prey pos:", tprey.pos())

        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)

        if captura_t(tpreds, tprey):
            print("presa apanhada!!!")

            finaldists = [toroidalDistance_coords(tpred.position(), tprey.position(), HEIGHT, WIDTH) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds

            #print("fitness:", 1)
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ (10*STEP))
            map.clearscreen()

            frames[0].save("gifs\\predatorTrialSuccess" + str(cont) +"_SignalsInd.gif",
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
    frames[0].save("gifs\\best_genomeTrialRun" + str(cont) +"_SignalsInd.gif",
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
    signals = [0 for n in range(n_preds)]

    for count in range(ticks): #(500 * 2 / 10) * (3/2) = 150
        newagentpos = []
        for npredsig in range(n_preds):#for pred, signal in zip (preds, signals):
            pred = preds[npredsig]
            sinaisOutrem =  copy.deepcopy(signals)
            del sinaisOutrem[npredsig]
            output, signal = ann_inputs_outputs_signal_I(pred, sinaisOutrem, prey, net)
            signals[npredsig] = signal
            #no simula por aqui para mudar a cor do agente de acordo com o sinal
            newagentpos.append((pred,pred_newpos(pred, output, STEP)))

        #pos da prey no fim da lista
        newagentpos.append((prey, prey_newpos(prey, preds, STEP)))
        
        #To make the prey not move use this line bellow instead of 196, the method function to make it move
        #newagentpos.append((prey, prey.get_coords()))
        
        #print("limpa movimentos")
        newagentmovements = limpamovimentos(newagentpos)
        #print("END limpa movimentos")
        
        #só se movem os agentes que não vão para a mesma casa nem que se atravessam
        for agent, agentnewpos in newagentmovements:
            agent.set_coords(agentnewpos[0], agentnewpos[1])


        #print("pred1_initialpos: ", pred1.get_initial_coords())
        #print("prey pos: ", prey.get_coords())

        #print("Media das distâncias ortogonais iniciais de todos os predadores à presa:", mediainidists)
        

        if captura(preds, prey):#if dist1 <= 40 or dist2 <= 40 or dist3 <= 40 or dist4 <= 40:

            finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), height, width) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds
        
        #print("Media das distâncias ortogonais finais de todos os predadores à presa:", mediafinaldists)

            print("presa apanhada!!!")
            #print("presa: ", prey.get_coords())
            print()
            print("fitness:", (2*(width + height) - mediafinaldists)/ (10*STEP))
            the_behaviour = [pred.get_coords() for pred in preds]
            #the_behaviour = tuple(finaldists)
            return ((2*(width + height) - mediafinaldists)/ (10*STEP)), the_behaviour # max threshold is 160 ((1600 - 0) / 10)

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
    the_behaviour = [pred.get_coords() for pred in preds]
    #print("the_behaviour", the_behaviour)
    #the_behaviour = tuple(finaldists)
    return (mediainidists - mediafinaldists) / (10*STEP), the_behaviour


# more constants

PREDS_captura = createpreds_parasimularchoque(STEP)
PREY_captura = prey_parasimularchoque()

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
    global GEN_N
    print("GEN_N", GEN_N)
    #genome_count = 0
    for genome_id, genome in genomes:
        #genome_count += 1
        #print("\nGENOME COUNT", genome_count )
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net, PREDS_DEF, PREYS_9, HEIGHT, WIDTH, TICKS)
        novelty_score = calculate_novelty(the_behaviour, NOVELTY_ARCHIVE, DIST)
        #print("fitness score:", fitness_result)
        #print("NOVELTY SCORE:", novelty_score)
        
        if novelty_score >= FITNESS_THRESHOLD:
            novelty_score = 17.9

        # Add behavior to the archive if it is sufficiently novel
        threshold = 0.1  # Adjust this threshold as needed
        if novelty_score >= threshold:
            NOVELTY_ARCHIVE.append(the_behaviour)

        if fitness_result >= 17 or GEN_N == MAX_N: # or genome_count == POP_N - 1: supposed to be for selecting the fitness score of the last generation instead of novelty when reaching the end of the experiment without the solution
            #print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        #elif GEN_N == MAX_N - 1:
        #    genome.fitness = fitness_result
        else:
            #print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score
    GEN_N +=1

# Wrapper function to include generation tracking
def eval_genomes_with_generation(genomes, config, generation):  
    # Access the current generation from the population
    eval_genomes(genomes, config, generation)

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

    #print("p.generation", p.generation)
    # Run for up to 500 generations.
    best_genome = p.run(eval_genomes, 500)#500

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'signal1', -4: 'signal2', -5:"signal3", 0:'Move_outputp', 1:'Signal'}
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
            else: 
                best_genome = run_experiment(config_path, genomeloadfile)

            print("best_genome.fitness:", best_genome.fitness)
            #to get the best genome out of the n resulting best genomes of the n experiments 
            if best_genome.fitness > best_of_the_bestGenomefitness:
                best_of_the_bestGenomefitness = best_genome.fitness
                best_of_the_bestGenome = best_genome
                
            # Assuming 'genome' is your NEAT genome object
            genome_path = 'storedgenomes\\goodgenomes_SignalInd.pkl'

            # Save the genome to a file
            with open("storedgenomes\\goodgenomes_SignalInd.pkl", "wb") as f:
                pickle.dump(best_genome, f)
                f.close()

            if i < n-1:
                #rawinput()
                userinput = str(input("want to carry on with the program?:(y/n) "))
                if userinput == "n":
                    break
    #keep the best genome of the n experimentations in a separate file
    best_genome_path = 'storedgenomes\\bestgenome_SignalInd.pkl'
    with open("storedgenomes\\bestgenome_SignalInd.pkl", "wb") as f:
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
    simula(net, PREDS_DEF, PREYS_9, PREYS_TEST, HEIGHT, WIDTH, TICKS)
    print("The END.")


#nrunexperiment(1)


#loading checkpoint for continuation
def runCheckpointExperiment(filename):

    config_path = os.path.join(local_dir, 'exercise.ini')

    restoredPopulation = neat.Checkpointer.restore_checkpoint(filename)

    # Add a stdout reporter to show progress in the terminal.
    restoredPopulation.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    restoredPopulation.add_reporter(stats)
    restoredPopulation.add_reporter(neat.Checkpointer(5, filename_prefix=os.path.join(out_dir, 'neat-checkpoint-')))

    best_genome = restoredPopulation.run(eval_genomes, 906)

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'offx2', -4: 'offy2', -5: 'offx3', -6: 'offy3', -7: 'offx4', -8: 'offy4', 0:'Move_outputp1', 1:'Move_outputp2', 2:'Move_outputp3', 3:'Move_outputp4'}
    visualize.draw_net(restoredPopulation.config, best_genome, True, node_names=node_names, directory=out_dir)

    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))

    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

    #keep the best genome of the n experimentations in a separate file
    best_genome_path = 'storedgenomes\\bestgenome_NoComTeam1o.pkl'
    with open("storedgenomes\\bestgenome_NoComTeam1o.pkl", "wb") as f:
            pickle.dump(best_genome, f)
            f.close()

    print("best_of_the_bestGenome.fitness", best_genome.fitness)
    print("end of regular experimentation!")
    print()

    print("simulate behavior of best genome on the trained set:", best_genome)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    simula(net, PREDS_DEF, PREYS_9, PREYS_DEF, HEIGHT, WIDTH, TICKS)


#checkpointfile = "out\\neat-checkpoint-94"
#runCheckpointExperiment(checkpointfile)

nrunexperiment(1, "storedgenomes\\goodgenomes_SignalInd.pkl")


#Pred1 = PREDS_captura[0]
#Pred2 = PREDS_captura[1]
#Pred3 = PREDS_captura[2]
#Pred4 = PREDS_captura[3]
#
#
#newagentpos = [(Pred1, (0, 0)), (Pred2,(STEP, 0)), (Pred3, (STEP*2, 0)), (Pred4,(STEP*3, 0)), (PREY_captura, (0, 0))]
#
#print(atravessaram((Pred1, (-2*STEP, 0)), PREDS_pos))
#
#problemas = True
#while problemas:
#
#    problemas = False
#    newagentpos2 = []
#    for ag, ag_pos in newagentpos:
#        if choca((ag, ag_pos), newagentpos) or atravessaram((ag, ag_pos), newagentpos):
#            problemas = True
#            newagentpos2.append((ag, ag.get_coords()))
#        else:
#            newagentpos2.append((ag, ag_pos))
#    newagentpos = newagentpos2
#print("PREDS_pos",newagentpos)
#print(choca((Pred1, (-STEP, 0)), PREDS_pos))
#print(captura(PREDS_captura, PREY_captura))