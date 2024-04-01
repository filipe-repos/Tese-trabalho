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

DIM = 2

DIST = 350
WIDTH = 350
HEIGHT = 350
STEP = 7
N_EVALS = 1
N_PREDS = 4
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
         x = math.copysign(max_width, x * -1)
    if abs(y) > max_height:
         y = math.copysign(max_height, y * -1)
    return x,y

#to use only in simulation with visualization        
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
    closestdistance = toroidalDistance_coords(preds[0].get_coords(), prey.get_coords(), WIDTH, HEIGHT)
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

#nova versão para objetos não turtle para não visualização
def captura_a(preds, prey):
    for p in preds:
        if prey.get_coords() == p.get_coords():
            return True
    return atravessaram_a(preds, prey)

def atravessaram_a(preds, prey):
    for p in preds:
        if p.get_old_coords() == prey.get_coords() and prey.get_old_coords() == p.get_coords():
            return True
    return False    


#preds_list= createpredators_obsolete(DIM, DIST, N_EVALS)
#prey = createPrey_obsolete(DIM, DIST)
preds_def = createpredators_bottom(HEIGHT, WIDTH, N_PREDS, STEP)
preys_def = createPreys(HEIGHT, WIDTH, preds_def, STEP, N_EVALS)
preys_test = createPreys(HEIGHT, WIDTH, preds_def, STEP, N_EVALS*10)

def ann_inputs_outputs(preds, prey, net):
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

def ann_inputs_outputs_t(tpreds, tprey, net):
    preds_coords = [p.position() for p in tpreds]
    preyx, preyy = tprey.position()
    input_data = []
    for x,y in preds_coords:
        print("\npredx: ", x, "preyx:", preyx)
        print("predy: ", y, "preyy:", preyy)
        offsetx = toroidalDistance1coord(x, preyx, WIDTH)
        offsety = toroidalDistance1coord(y, preyy, HEIGHT)
        norm_offsetx = ((offsetx/ WIDTH) +1 )/2
        norm_offsety = ((offsety/ HEIGHT) +1 )/2
        print("offsetx: ", offsetx, " ; norm_offsetx", norm_offsetx)
        print("offsety: ", offsety, " ; norm_offsety", norm_offsety)
        input_data.append(norm_offsetx)
        input_data.append(norm_offsety)
    #print("INPUTS!:", input_data1)
    return net.activate(input_data)


#simula
def simula(net, dim):
    cont = 0
    for prey in preys_def:
        cont+=1
        print("simulação",cont)
        simula1(net,copy.deepcopy(preds_def), copy.deepcopy(prey), dim, TICKS)
    #testing best genome in new situations with new prey positions
    print("testing best genome in new situations with new prey positions")
    for prey in preys_test:
        simula1(net,copy.deepcopy(preds_def), copy.deepcopy(prey), dim)

def simula1(net, preds, prey, dist, ticks):
    
    preds = copy.deepcopy(preds)#preds

    prey = copy.deepcopy(prey)
    prey_coords = prey.get_coords()

    colors = ["yellow", "orange", "red", "black", "yellow", "orange", "red", "black"]
    n_preds = len(preds)

    map = turtle.Screen()
    map.screensize(dist, dist)
    map.bgcolor("lightgreen")    # set the window background color
    #map.tracer(0, 0)             # to make map not display anything making the execution much faster
    tpreds = []

    for pred, color in zip(preds, colors):
        tpred= turtle_agent(pred.get_coords(), color)
        tpreds.append(tpred)

    tprey = turtle_agent(prey_coords, "blue", "turtle")

    frames = []
    window = gw.getWindowsWithTitle("Python Turtle Graphics")[0]
    com,larg =window.size
    window.moveTo(10,10)

    #while count <= ((HEIGHT*2) / STEP) * 1.5: #(500 * 2 / 10) * (3/2) = 150 (350 *2) /7)) *1.5
    for count in range(ticks):
        count +=1
        
        outputs = ann_inputs_outputs_t(tpreds, tprey, net)
        for tpred, output in zip(tpreds, outputs):
            tpred_move(tpred, output, STEP)

        #print("pred new coords: ", tpred.position())
        #To make the prey not move commented the method function to make it move
        #tprey_move(tprey, (tpred1, tpred2, tpred3, tpred4), STEP)

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

    inidists = [toroidalDistance_coords(tpred.initial_coords, tprey.position(), HEIGHT, WIDTH) for tpred in tpreds]
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


local_dir = os.path.dirname(__file__)
# The directory to store outputs
out_dir = os.path.join(local_dir, 'out')

input_data = ([5,5], [5,6], [10,10], [10,11], [20,20], [19,20], [1,20])

#calculo da média dos fitness das várias avaliações de cada rede neuronal numa lista de preys
def eval_fitness(net):
    the_fitness= 0
    #cycle_count = 0
    #print([p.get_coords() for p in preys_def])
    for prey in preys_def:
        #cycle_count += 1
        #print("CYCLE:", cycle_count)
        f1 = eval_fitness1(net, prey, TICKS)
        #print("eval1", f1)
        the_fitness += f1
    #print("the summed fitness:", the_fitness, "the number of experiments per genome:", len(preys_def))
    return the_fitness/ len(preys_def)


#Urgent to change 
def eval_fitness1(net, theprey, ticks):
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
    for count in range(ticks): #(500 * 2 / 10) * (3/2) = 150
        
        outputs = ann_inputs_outputs(preds, prey, net)
        for pred, output in zip(preds, outputs):
            pred_move(pred, output, STEP)
            #print("pred new coords: ", pred.get_coords())


        #To make the prey not move commented the method function to make it move
        #prey_move(prey, preds, STEP)

        #print("pred1_initialpos: ", pred1.get_initial_coords())
        #print("prey pos: ", prey.get_coords())

        #print("Media das distâncias ortogonais iniciais de todos os predadores à presa:", mediainidists)
        

        if captura_a(preds, prey):#if dist1 <= 40 or dist2 <= 40 or dist3 <= 40 or dist4 <= 40:

            finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), HEIGHT, WIDTH) for pred in preds]
            mediafinaldists = sum(finaldists) / n_preds
        
        #print("Media das distâncias ortogonais finais de todos os predadores à presa:", mediafinaldists)

            print("presa apanhada!!!")
            #print("presa: ", prey.get_coords())
            print()
            print("fitness:", (2*(WIDTH + HEIGHT) - mediafinaldists)/ 10)
            return ((2*(WIDTH + HEIGHT) - mediafinaldists)/ 10) # max threshold is 160 ((1600 - 0) / 10)

    inidists = [toroidalDistance_coords(pred.get_initial_coords(), prey.get_coords(), HEIGHT, WIDTH) for pred in preds]
    mediainidists = sum(inidists) / n_preds

    finaldists = [toroidalDistance_coords(pred.get_coords(), prey.get_coords(), HEIGHT, WIDTH) for pred in preds]
    mediafinaldists = sum(finaldists) / n_preds
    #print("fitness:",1/(dist1 + dist2 + dist3 + dist4))
    #print("fitness:", (mediainidists - mediafinaldists) / 10)
    return (mediainidists- mediafinaldists) / 10



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
    best_genome = p.run(eval_genomes, 500)#300

    # Display the best genome among generations.
    print('\nBest genome:\n{!s}'.format(best_genome))

    # Visualize the experiment results
    node_names = {-1:'offx1', -2: 'offy1', -3: 'offx2', -4: 'offy2', -5: 'offx3', -6: 'offy3', -7: 'offx4', -8: 'offy4', 0:'Move_output?'}
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

    # Assuming 'genome' is your NEAT genome object
    genome_path = 'genome.pkl'

    # Save the genome to a file
    with open("bestgenome.pkl", "wb") as f:
        pickle.dump(best_genome, f)
        f.close()

    # Later, you can load the genome from the file
    #with open(genome_path, 'r') as f:
    #    loaded_genome = neat.Genome()
    #    loaded_genome.parse(f)
    print("end of regular experimentation!")
    print()
    print("simulate behavior of best genome:", best_genome)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    simula(net, DIST)
    print("The END.")
