#
# This file provides source code of the predator prey coordenation experiment using on NEAT-Python library
#

import turtle


# The Python standard library import
import os
import shutil
import random
import numpy as np
# The NEAT-Python library imports
import neat
# The helper used to visualize experiment results
import visualize
import copy
from collections import deque

from agent import *
from PIL import Image, ImageGrab
import pygetwindow as gw


HEIGHT = 100
WIDTH = 350
DIM = 10
STEP = WIDTH/10
N_EVALS = 10
# Define a global novelty archive
NOVELTY_ARCHIVE = deque(maxlen=500)  # Set a limit to the archive size

def createpreds(dim, n):
    pred_list = []
    for i in range(n):
        rand1= random.randint(1,DIM)
        rand2= random.randint(1,DIM)
        pred1 = Predator(rand1)
        pred2 = Predator(rand2)
        pred_list.append((pred1,pred2))
    return pred_list

def simula(net, dim, height, width, step):
    cont = 0
    fitnesses = 0
    for preds in preds_list:
        cont+=1
        print("simulação",cont)
        fitness = simula1(net,copy.deepcopy(preds), dim, height, width, step, cont)
        fitnesses += fitness
    print("a média de fitness dos ensaios é:",fitnesses/cont)

def simula1(net, preds, dim, height, width, step, cont):

    predator1, predator2 = preds
    count = 0
    signal1 = 0
    signal2 = 0

    #turtledisplay
    map = turtle.Screen()
    map.screensize(height, width)
    map.bgcolor("lightgreen")    # set the window background color
    #map.tracer(0, 0)             # to make map not display anything making the execution much faster
    map.delay(30)

    pred1, pred2 = preds
    display_preds(predator1, predator2, dim)
    
    #turtledisplay
    positionx1 = -(pred1.get_distanceToPrey() * width / 10)
    positionx2 = pred2.get_distanceToPrey() * width / 10
    tpred1 = turtle_agent((positionx1, 0), "black")
    tpred2 = turtle_agent((positionx2, 0), "black")
    tpred2.setheading(180)
    tprey = turtle_agent((0,0), "blue", "turtle")
    #turtledisplay
    frames = []
    window = gw.getWindowsWithTitle("Python Turtle Graphics")[0]
    com,larg =window.size
    window.moveTo(10,10)
    
    image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
    frames.append(image)
    
    while count <= dim*2:
        count +=1
        d1 = predator1.get_distanceToPrey()/DIM #, predator2.get_distanceToPrey()]
        d2 = predator2.get_distanceToPrey()/DIM

        input1 = [d1,signal2]
        input2 = [d2,signal1]
        
        output1, signal1 = tuple(net.activate(input1))
        output2, signal2 = tuple(net.activate(input2))

        predator1.move(round(output1))
        #turtledisplay
        color1 = signal1
        tpred1.fillcolor(color1, 0, 0)
        turtlemove(tpred1, output1, step)

        predator2.move(round(output2))
        #turtledisplay
        color2 = signal2
        tpred2.fillcolor(color2, 0, 0)
        turtlemove(tpred2, output2, step)

        display_preds_s(predator1, predator2, signal1, signal2, dim)

        #turtledisplay
        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)
        image = ImageGrab.grab(bbox=(10, 10, 10+com, 10+larg))
        frames.append(image)

        dist1 = predator1.get_distanceToPrey()
        dist2 = predator2.get_distanceToPrey()
        if (dist1 + dist2) == 0:
            #print("entrei")
            print("fitness:", 1)
            #turtledisplay
            map.clearscreen()
            frames[0].save("gifs\\1stExpTrial" + str(cont) +"_EvolvingComunicationModel_Success.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=1)  # Set loop to 0 for infinite loop
            return 1

        if (dist1 == 0 or dist2 == 0):
            print("fitness:", 0)
            #turtledisplay
            map.clearscreen()
            frames[0].save("gifs\\1stExpTrial" + str(cont) +"_EvolvingComunicationModel_Fail.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=1)  # Set loop to 0 for infinite loop
            return 1

    print("fitness:",1/(dist1 + dist2))
    #turtledisplay
    map.clearscreen()
    frames[0].save("gifs\\1stExpTrial" + str(cont) +"_EvolvingComunicationModel_Fail.gif",
               save_all=True,
               append_images=frames[1:],
               duration=100,  # Set the duration for each frame in milliseconds
               loop=1)  # Set loop to 0 for infinite loop
    return 1/(dist1 + dist2)


preds_list= createpreds(DIM, N_EVALS)

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

input_data = ([5,5], [5,6], [10,10], [10,11], [20,20], [19,20], [1,20])

def eval_fitness(net):
    the_fitness= 0
    behaviours = []
    the_behaviour = (0,0)
    for preds in preds_list:
        f1, behaviour = eval_fitness1(net, copy.deepcopy(preds))
        behaviours.append(behaviour)

        #print("eval1", f1)
        the_fitness += f1
        the_behaviour += behaviour

    return the_fitness/ len(preds_list), behaviours#tuple(x / len(preds_list) for x in behaviours) #the_behaviour/ len(preds_list)

def eval_fitness1(net, preds):
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

    predator1, predator2 = preds
    count = 0
    sinal1=0
    sinal2=0
    while count <= DIM*2:
        count +=1
        d1 = predator1.get_distanceToPrey()/DIM #, predator2.get_distanceToPrey()]
        d2 = predator2.get_distanceToPrey()/DIM

        #input1 = [d1,d2]
        #input2 = [d2,d1]
        input1 = [d1,sinal2]
        input2 = [d2,sinal1]

        #print("input[0]",input[0])
        #print("input[1]",input[1])
        #print("(input)[1]",(input)[1])
        output1, sinal1= tuple(net.activate(input1)) 
        output2, sinal2= tuple(net.activate(input2))

        #output1 = int(round(net.activate(input1)[0]))
        #output2 = int(round(net.activate(input2)[0]))
        
        #output1 = net.activate(input_data1)[0]
        #output2 = net.activate(input_data2)[0]

        # Assign fitness based on the output
        #print("output1 =", output1)
        #print("output2 =", output2)
        #print("predator1 distance to prey:", predator1.get_distanceToPrey())
        #print("predator2 distance to prey:", predator2.get_distanceToPrey())

        predator1.move(round(output1))
        predator2.move(round(output2))

        dist1 = predator1.get_distanceToPrey()
        dist2 = predator2.get_distanceToPrey()
        #print("new predator1 distance to prey:", dist1)
        #print("new predator2 distance to prey:", dist2)
        #print("soma das novas distâncias:", (dist1 + dist2))

        if (dist1 + dist2) == 0:
            #print("entrei")
            fitness = 1
            #behaviour = 1
            behaviour = (dist1, dist2)
            return fitness, behaviour
        
        if (dist1 == 0 or dist2 == 0):
            fitness = 0
            #behaviour = 1/(dist1 + dist2+ 5)
            behaviour = (dist1, dist2)
            return fitness, behaviour

        #else:
        #    fitness = 1/(predator1.get_distanceToPrey() + predator2.get_distanceToPrey())
        #display_preds(predator1, predator2)
    #print("end of a cycle!!!")
    fitness = 1/(dist1 + dist2)
    #behaviour =  1/(dist1 + dist2)
    behaviour = (dist1, dist2)
    return fitness, behaviour

def display_preds(predator1, predator2, dim):
    a = predator1.get_distanceToPrey()
    b = predator2.get_distanceToPrey()
    print("ilustration of environment and distance to predator:",a,b)
    for i in range(dim, 0, -1):
        if i == a:
            print(1,end = " ")
        else:
            print(".",end = " ")
    print("*",end = " ")
    for i in range(1, dim+1):
        if i == b:
            print(2,end = " ")
        else:
            print(".",end = " ")
    print()

def display_preds_s(predator1, predator2, signal1, signal2, dim):
    a = predator1.get_distanceToPrey()
    b = predator2.get_distanceToPrey()
    print("ilustration of environment and distance to predator:",a,b)
    for i in range(dim, 0, -1):
        if i == a:
            print(1,end = " ")
        else:
            print(".",end = " ")

    print("*",end = " ")

    for i in range(1, dim+1):
        if i == b:
            print(2,end = " ")
        else:
            print(".",end = " ")
    print()
    print("signal1 sent=",signal1, end=" ")
    print("signal2 sent=",signal2)
    print()
    
def calculate_novelty(behavior, archive):
    # Calculate novelty as the average distance to the k-nearest neighbors in the archive
    k = min(10, len(archive))  # Use up to 10 neighbors
    if k == 0:
        return float('0.1')  # 0.1 novelty if archive is empty

    distances = sorted([np.linalg.norm(np.array(behavior) - np.array(b)) for b in archive])
    novelty_score = np.mean(distances[:k])/(DIM*2)
    return novelty_score

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
    for genome_id, genome in genomes:
        print("GENOME ", genome_id)
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_result, the_behaviour = eval_fitness(net)
        novelty_score = calculate_novelty(the_behaviour, NOVELTY_ARCHIVE)
        print("fitness score:", fitness_result)
        print("NOVELTY SCORE:", novelty_score)
        
        # Add behavior to the archive if it is sufficiently novel
        threshold = 0.1  # Adjust this threshold as needed
        if novelty_score >= threshold:
            NOVELTY_ARCHIVE.append(the_behaviour)

        if fitness_result == 1:
            print("solution found, fitness used!\n")
            genome.fitness = fitness_result
        else:
            print("solution not found, novelty score used for selection!\n")
            genome.fitness = novelty_score

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

    
    #Cicle to check every individual of population fitness
    #for ind in p.population.values():
        #print("ind",ind)
        #net = neat.nn.FeedForwardNetwork.create(ind, config)
        #print("fitness",eval_fitness(net))
    #print("fim do ciclo!!!!")

    # Run for up to 300 generations.
    best_genome = p.run(eval_genomes, 100)#300

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))
    # Show output of the most fit genome against training data.
    #print('\nOutput:')
    #net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    #simula(net, DIM)

    # Check if the best genome is an adequate XOR solver
    #best_genome_fitness = eval_fitness(net)
    #if best_genome_fitness > config.fitness_threshold:
    #    print("\n\nSUCCESS: The Predator problem  solver found!!!")
    #else:
    #    print("\n\nFAILURE: Failed to find Predator problem solver!!!")

    # Visualize the experiment results
    node_names = {-1:'distanceToPrey1', -2: 'signal2 input', 0:'Move output', 1:'signal1 output'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

    return best_genome


def clean_output():
    if os.path.isdir(out_dir):
        # remove files from previous run
        shutil.rmtree(out_dir)

    # create the output directory
    os.makedirs(out_dir, exist_ok=False)

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    config_path = os.path.join(local_dir, 'firstexercise.ini')

    # Clean results of previous run if any or init the output directory
    clean_output()
    print("BEGINNING!")
    # Run the experiment
    best_genome = run_experiment(config_path)
    print("end of regular experimentation!")
    print()
    print("simulate behavior of best genome:", best_genome)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    #simula(net, DIM)
    simula(net, DIM, HEIGHT, WIDTH, STEP)
    print("The END.")
    #predator1 = Predator(1)
    #predator2 = Predator(10)
    #display_preds(predator1, predator2, 10)
