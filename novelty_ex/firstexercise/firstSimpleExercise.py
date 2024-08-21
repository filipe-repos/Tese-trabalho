#
# This file provides source code of the predator prey coordenation experiment using on NEAT-Python library
#

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

from agent import Predator

DIM = 10
N_EVALS = 10
def createpreds(dim, n):
    pred_list = []
    for i in range(n):
        rand1= random.randint(1,DIM)
        rand2= random.randint(1,DIM)
        pred1 = Predator(rand1)
        pred2 = Predator(rand2)
        pred_list.append((pred1,pred2))
    return pred_list

def simula(net, dim):
    cont = 0
    fitnesses = 0
    for preds in preds_list:
        cont+=1
        print("simulação",cont)
        fitness = simula1(net,copy.deepcopy(preds), dim)
        fitnesses += fitness
    print("a média de fitness dos ensaios é:",fitnesses/cont)

def simula1(net, preds, dim):
    the_fitness = 0.0

    predator1, predator2 = preds
    count = 0
    display_preds(predator1, predator2, dim)
    while count <= dim*2:
        count +=1
        d1 = predator1.get_distanceToPrey()/DIM #, predator2.get_distanceToPrey()]
        d2 = predator2.get_distanceToPrey()/DIM

        input1 = [d1,d2]
        input2 = [d2,d1]
        output1 = int(round(net.activate(input1)[0]))
        output2 = int(round(net.activate(input2)[0]))

        predator1.move(output1)
        predator2.move(output2)

        display_preds(predator1, predator2, dim)

        dist1 = predator1.get_distanceToPrey()
        dist2 = predator2.get_distanceToPrey()
        if (dist1 + dist2) == 0:
            #print("entrei")
            print("fitness:", 1)
            return 1

        if (dist1 == 0 or dist2 == 0):
            print("fitness:", 0)
            return 0

    print("fitness:",1/(dist1 + dist2))
    return 1/(dist1 + dist2)
    


preds_list= createpreds(DIM, N_EVALS)

# The current working directory
local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

input_data = ([5,5], [5,6], [10,10], [10,11], [20,20], [19,20], [1,20])

def eval_fitness(net):
    the_fitness= 0
    for preds in preds_list:
        f1 = eval_fitness1(net, copy.deepcopy(preds))
        #print("eval1", f1)
        the_fitness += f1

    return the_fitness/ len(preds_list)

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

        inputsp1 = [d1,d2]
        inputsp2 = [d2,d1]
        #input1 = [d1,sinal2]
        #input2 = [d2,sinal1]

        #print("input[0]",input[0])
        #print("input[1]",input[1])
        #print("(input)[1]",(input)[1])
        #output1, sinal1= tuple(net.activate(input1)) 
        #output2, sinal2= tuple(net.activate(input2))

        output1 = int(round(net.activate(inputsp1)[0]))
        output2 = int(round(net.activate(inputsp2)[0]))
        
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
            return 1
        
        if (dist1 == 0 or dist2 == 0):
            return 0

        #else:
        #    fitness = 1/(predator1.get_distanceToPrey() + predator2.get_distanceToPrey())
        #display_preds(predator1, predator2)
    #print("end of a cycle!!!")
    return  1/(dist1 + dist2)

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
    

def eval_fitness2(net):
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
    for distance_to_objective in input_data:  # Distance from 0 to 100
        predator1 = Predator(distance_to_objective[0], distance_to_objective[1])
        predator2 = Predator(distance_to_objective[1], distance_to_objective[0])

        input_data1 = [distance_to_objective[0] / 100, distance_to_objective[1]]
        input_data2 = [distance_to_objective[1] / 100, distance_to_objective[0]]
        output1 = net.activate(input_data1)[0]
        output2 = net.activate(input_data2)[0]

        # Assign fitness based on the output
        if output1 > 0.5:
            predator1.move() 
        if output2 > 0.5:
            predator2.move()

        if predator1.get_distanceToPrey() == 0 and predator1.get_distanceToPrey() == 0:
            fitness = 1
        elif predator1.get_distanceToPrey() == 0 and predator1.get_distanceToPrey() != 0:
            fitness = 0
        elif predator1.get_distanceToPrey() != 0 and predator1.get_distanceToPrey() == 0:
            fitness = 0
        else:
            fitness = 1/(predator1.get_distanceToPrey() + predator2.get_distanceToPrey())
        
        the_fitness += fitness
    return the_fitness

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
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net)

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
    #for ind in p.population.values():
        #print("ind",ind)
        #net = neat.nn.FeedForwardNetwork.create(ind, config)
        #print("fitness",eval_fitness(net))

    print("fim do ciclo!!!!")
    best_genome = p.run(eval_genomes, 100)#300

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))
    # Show output of the most fit genome against training data.
    #print('\nOutput:')
    #net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    #for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # Check if the best genome is an adequate XOR solver
    #best_genome_fitness = eval_fitness(net)
    #if best_genome_fitness > config.fitness_threshold:
    #    print("\n\nSUCCESS: The XOR problem solver found!!!")
    #else:
    #    print("\n\nFAILURE: Failed to find XOR problem solver!!!")

    # Visualize the experiment results
    node_names = {-1:'distanceToPrey1', -2: 'distanceToPrey2', 0:'Movement Output'}
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
    #print("aaaaaaaaaaaaaaaaaaaaaaahhhhhhhhhhhhhhhhhhhh")
    # Run the experiment
    best_genome = run_experiment(config_path)
    #print("AAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHH")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    simula(net, DIM)
    #print("AAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHH")
    #predator1 = Predator(1)
    #predator2 = Predator(10)
    #display_preds(predator1, predator2, 10)
