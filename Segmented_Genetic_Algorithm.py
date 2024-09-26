import random
import heapq
import math
import copy
import time


#reads the node locations from the input file
#return a list with the number of locations at index 0 and locations at index 1
def readFile(fileName):
    f = open(fileName, "r")
    size = int(f.readline())
    nodes = {}
    for i in range(size):
        j = f.readline().split()
        nodes[i] = (int(j[0]), int(j[1]), int(j[2]))
    return [size, nodes]


#generate initial population of size//segments for each segment
#generates multiple random populations and selects the top individuals from each
def gen_init_pop(nodes, pop, size, pool, segments, anchor):
    initPop = []
    for i in range(4):
        partialPop = []
        for i in range(pop):
            path = anchor + random.sample(pool, size//segments)
            partialPop.append(path)
        partialPop = cull(nodes, partialPop)[1]
        initPop.extend(cull(nodes, partialPop)[1])
    return initPop


#generate the initial population for the final segment. uses rest of pool instead of a set size
def gen_final_pop(nodes, pop, pool, anchor):
    initPop = []
    for i in range(4):
        partialPop = []
        for i in range(pop):
            spool = list(pool)
            random.shuffle(spool)
            path = anchor + spool
            partialPop.append(path)
        partialPop = cull(nodes, partialPop)[1]
        initPop.extend(cull(nodes, partialPop)[1])
    return initPop

#get the circular distance of a path starting and ending at the node at index 0
def get_distance(nodes, path):
    total_distance = 0
    for i in range(-1, len(path) - 1):
        total_distance += distance(nodes[path[i]], nodes[path[i + 1]])
    return total_distance

#get the length of a path starting at index 0 and ending at the last index
def get_segment_length(nodes, path):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance(nodes[path[i]], nodes[path[i + 1]])
    return total_distance


#implementing dist because math.dist isn't available ein python 3.7 :(
def distance(node1, node2):
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2 + (node1[2] - node2[2]) ** 2)


#cull the population, compare two individuals together and keep the shorter one until we are left with half the population
def cull(nodes, old_pop):
    culled = []
    best_path = [float('inf'), []]
    random.shuffle(old_pop)
    halfway = len(old_pop) // 2
    for i in range(halfway):
        dist1 = get_segment_length(nodes, old_pop[i])
        dist2 = get_segment_length(nodes, old_pop[i + halfway])
        if dist1 > dist2:
            if dist2 < best_path[0]:
                best_path = [dist2, old_pop[i + halfway]]
            culled.append(old_pop[i + halfway])
        else:
            if dist1 < best_path[0]:
                best_path = [dist1, old_pop[i]]
            culled.append(old_pop[i])
    return [best_path, culled]


#crossover parent function, choose a daddy and a mommy path to create offspring
def crossover(pop):
    babies = []
    halfway = len(pop) // 2
    for i in range(halfway):
        dad = pop[i]
        mom = pop[halfway + i]
        babies.extend(make_babies(dad, mom))
    return babies


#use the daddy and mommy paths to create offspring by randomly selecting a section from one parent and inserting it into
#the other one while removing duplicates from the one being inserted into. Also returns keeps the parents in the
#population in case the children are disappointments.
def make_babies(dad, mom):
    quadruplets = []

    start = random.randint(1, len(dad) - 1)
    finish = random.randint(start, len(dad))
    dad_genes = dad[start:finish]
    mom_genes = [j for j in mom if j not in dad_genes]
    boy = mom_genes[:start] + dad_genes + mom_genes[start:]

    start = random.randint(1, len(mom) - 1)
    finish = random.randint(start, len(mom))
    mom_genes = mom[start:finish]
    dad_genes = [k for k in dad if k not in mom_genes]
    girl = dad_genes[:start] + mom_genes + dad_genes[start:]


    quadruplets.append(dad)
    quadruplets.append(mom)
    quadruplets.append(boy)
    quadruplets.append(girl)
    return quadruplets


#add mutations to generation to inject more variance and avoid converging on local maxima
def mutate(population):
    num_mutations = len(population) // 100
    for i in range(num_mutations):
        lucky = population[random.randint(0, len(population) - 1)]
        index1 = random.randint(1, len(lucky) - 1)
        index2 = random.randint(1, len(lucky) - 1)
        lucky[index1], lucky[index2] = lucky[index2], lucky[index1]


#main genetic algorithm function that ties everything together. Splits the path into segments of SEGMENT_LENGTH and runs
#the genetic algorithm on each segment. Removes elements already used in previous segments from the pool and returns
#the combined path with the shortest path of each segment. Also chooses and anchor each time to ensure that the path
#can be seamlessly connected.
def genetic_alg(nodes, POP_SIZE, size, generations, segments):
    anchor = [0]
    pool = list(range(1, size))
    combined_segments = [0]
    for i in range(segments-1):
        current_population = gen_init_pop(nodes, POP_SIZE, size, pool, segments, anchor)
        best_path = [float('inf'), []]
        for i in range(generations):
            culled = cull(nodes, current_population)
            if best_path[0] > culled[0][0]:
                best_path = culled[0]
            mutate(current_population)
            current_population = crossover(culled[1])
        for i in current_population:
            p = get_segment_length(nodes, i)
            if p < best_path[0]:
                best_path = [p, i]
        best_segment = best_path[1]
        combined_segments.extend(best_segment[1:])
        anchor = [best_segment[-1]]
        for j in best_segment[1:]:
            pool.remove(j)
    current_population = gen_final_pop(nodes, POP_SIZE, pool, anchor)
    best_path = [float('inf'), []]
    for i in range(generations):
        culled = cull(nodes, current_population)
        if best_path[0] > culled[0][0]:
            best_path = culled[0]
        mutate(current_population)
        current_population = crossover(culled[1])
    for i in current_population:
        p = get_segment_length(nodes, i)
        if p < best_path[0]:
            best_path = [p, i]
    best_segment = best_path[1]
    combined_segments.extend(best_segment[1:])

    return combined_segments


#Writes the result to and output file
def write_output(nodes, result):
    f = open("output.txt", "w")
    f.write(str(result[0]))
    for i in result[1]:
        f.write("\n" + str(nodes[i][0]) + " " + str(nodes[i][1]) + " " + str(nodes[i][2]))
    f.write("\n" + str(nodes[result[1][0]][0]) + " " + str(nodes[result[1][0]][1]) + " " + str(nodes[result[1][0]][2]))


#Generate an initial population for when the size <= SEGMENT_LENGTH and we only have 1 segment
def gen_small_init_pop(nodes, pop, size):
    initPop = []
    for i in range(4):
        partialPop = []
        for i in range(pop):
            path = list(range(size))
            random.shuffle(path)
            partialPop.append(path)
        partialPop = cull(nodes, partialPop)[1]
        initPop.extend(cull(nodes, partialPop)[1])
    return initPop


#A small genetic algorithm for when we only have 1 segment, just runs the genetic algorithm on the entire population
def small_genetic_alg(nodes, initial_population, generations):
    current_population = initial_population
    best_path = [float('inf'), []]
    for i in range(generations):
        culled = cull(nodes, current_population)
        if best_path[0] > culled[0][0]:
            best_path = culled[0]
        current_population = small_crossover(culled[1])
    for i in current_population:
        p = get_distance(nodes, i)
        if p < best_path[0]:
            best_path = [p, i]

    return best_path


#Crossover function for when we only have 1 segment
def small_crossover(pop):
    babies = []
    halfway = len(pop) // 2
    for i in range(halfway):
        dad = pop[i]
        mom = pop[halfway + i]
        babies.extend(small_make_babies(dad, mom))
    return babies


#Same as the other make_babies function except we don't account for the anchor
def small_make_babies(dad, mom):
    quadruplets = []

    start = random.randint(0, len(dad) - 1)
    finish = random.randint(start, len(dad))
    dad_genes = dad[start:finish]
    mom_genes = [j for j in mom if j not in dad_genes]
    boy = mom_genes[:start] + dad_genes + mom_genes[start:]

    start = random.randint(0, len(mom) - 1)
    finish = random.randint(start, len(mom))
    mom_genes = mom[start:finish]
    dad_genes = [k for k in dad if k not in mom_genes]
    girl = dad_genes[:start] + mom_genes + dad_genes[start:]

    quadruplets.append(dad)
    quadruplets.append(mom)
    quadruplets.append(boy)
    quadruplets.append(girl)
    return quadruplets



if __name__ == '__main__':

    tstart = time.time()
    #read the input file and get the number of nodes + the list of all nodes
    r = readFile('input1.txt')

    #set size equal to the first line from the function
    size = r[0]
    # turn nodes into a dict for easier access
    nodes = r[1]

    #set population size, number of generations, and the size of each segment, calculate how many segments we need
    POP_SIZE = 2500
    GENERATIONS = 150
    SEGMENT_SIZE = 50
    SEGMENTS = size//50

    #if we only have 1 segment, run the smaller algorithm, otherwise run the segmented algorithm
    if SEGMENTS == 1:

        #smaller algorithm runs faster so we can afford to increase the number of generations
        GENERATIONS = 250
        initial_pop = gen_small_init_pop(nodes, POP_SIZE, size)

        # run genetic algorithm
        result = small_genetic_alg(nodes, initial_pop, GENERATIONS)
        # write the result to an output file
        write_output(nodes, result)
    else:
        #run genetic algorithm
        best_path = genetic_alg(nodes, POP_SIZE, size, GENERATIONS, SEGMENTS)

        result = [get_distance(nodes, best_path), best_path]
        #write the result to an output file
        write_output(nodes, result)

    tend = time.time()
    print(tend - tstart)