import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.7,
            linearity=-0.5,
            solvability=1.0,
            

        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                if(y == 14):
                    if random.random() < 0.01:
                        if(genome[y][x] == "-"):
                            genome[y][x] = "E"
                
                if (y >= 8 and y <= 17):
                    if random.random() < 0.018:
                        if(genome[y][x] == "-"):
                            genome[y][x] = "o"


                if(genome[y][x] == "?"):
                    if random.random() < 0.3:
                        roll = random.randrange(1,4)
                        if(roll == 1):
                            genome[y][x] = "B"

                if(genome[y][x] == "B"):
                    if random.random() < 0.3:
                        roll = random.randrange(1,4)
                        if(roll == 1):
                            genome[y][x] = "?"
    

                if(genome[y][x] == "M"):
                    if random.random() < 0.25:
                        roll = random.randrange(1,3)
                        if(roll == 1):
                            genome[y][x] = "?"
                        else:
                            genome[y][x] = "B"

        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        left = 1
        right = width - 1

        for y in range(height):
            for x in range(left, right):
                if random.random() < 0.5:  # Uniform crossover
                    new_genome[y][x] = other.genome[y][x]

        # Mutate the child
        mutated_genome = self.mutate(new_genome)
            # do mutation; note we're returning the Individual_Grid object directly
        return Individual_Grid(new_genome)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        return self.genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf

class Individual_DE(object):
    __slots__ = ["genome", "_fitness", "_level"]

    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # Example penalty if too many stairs:
        if len([de for de in self.genome if de[1] == "6_stairs"]) > 5:
            penalties -= 2

        self._fitness = sum(coefficients[m] * measurements[m] for m in coefficients) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        if random.random() < 0.1 and len(new_genome) > 0:
            idx = random.randint(0, len(new_genome) - 1)
            de = new_genome[idx]
            new_genome.pop(idx)
        return new_genome

    def generate_children(self, other):
        if len(self.genome) == 0 or len(other.genome) == 0:
            return copy.deepcopy(self), copy.deepcopy(other)

        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)

        a_part = self.genome[:pa]
        b_part = other.genome[pb:]
        ga = a_part + b_part

        b_part = other.genome[:pb]
        a_part = self.genome[pa:]
        gb = b_part + a_part

        # Mutate children
        ga = self.mutate(ga)
        gb = self.mutate(gb)

        # Return **two** separate `Individual_DE` objects instead of a tuple
        return Individual_DE(ga), Individual_DE(gb)

    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda d: (d[1], d[0], d)):
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    # (x, "4_block", y, breakable)
                    y = de[2]
                    breakable = de[3]
                    # place a 'B' if breakable else 'X'
                    base[y][x] = "B" if breakable else "X"

                elif de_type == "5_qblock":
                    # (x, "5_qblock", y, has_powerup)
                    y = de[2]
                    has_powerup = de[3]  # bool
                    base[y][x] = "M" if has_powerup else "?"

                elif de_type == "3_coin":
                    # (x, "3_coin", y)
                    y = de[2]
                    base[y][x] = "o"

                elif de_type == "7_pipe":
                    # (x, "7_pipe", h)
                    h = de[2]
                    # Put pipe top at row = height - h - 1
                    top_row = height - h - 1
                    if 0 <= top_row < height:
                        base[top_row][x] = "T"
                    # Fill the pipe below that top row
                    for row in range(top_row + 1, height):
                        if 0 <= row < height:
                            base[row][x] = "|"

                elif de_type == "0_hole":
                    # (x, "0_hole", w)
                    w = de[2]
                    # Make a hole of width w in the floor
                    for dx in range(w):
                        xx = x + dx
                        if 1 <= xx < width - 1:
                            base[height - 1][xx] = "-"

                elif de_type == "6_stairs":
                    # (x, "6_stairs", h, dx)
                    h_stairs = de[2]
                    direction = de[3]  # -1 or +1
                    for step in range(1, h_stairs + 1):
                        real_x = x + step * direction
                        row = height - 1 - step
                        if 0 <= real_x < width and 0 <= row < height:
                            base[row][real_x] = "X"

                elif de_type == "1_platform":
                    # (x, "1_platform", w, y, madeof)
                    w_plat = de[2]
                    y_plat = de[3]
                    tile = de[4]  # "?", "X", or "B"
                    for dx in range(w_plat):
                        xx = x + dx
                        if 0 <= xx < width and 0 <= y_plat < height:
                            base[y_plat][xx] = tile

                elif de_type == "2_enemy":
                    # (x, "2_enemy")
                    # place an enemy somewhere near the floor (or at height-2)
                    if 0 <= x < width:
                        base[height - 2][x] = "E"

            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        elt_count = random.randint(8, 128)
        g = []
        for _ in range(elt_count):
            # Pick a random design element from the skeleton's set:
            g.append(random.choice([
                (random.randint(1, width - 2), "0_hole", random.randint(1, 8)),
                (random.randint(1, width - 2), "1_platform", random.randint(1, 8),
                 random.randint(0, height - 1), random.choice(["?", "X", "B"])),
                (random.randint(1, width - 2), "2_enemy"),
                (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
                (random.randint(1, width - 2), "4_block", random.randint(0, height - 1),
                 random.choice([True, False])),
                (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1),
                 random.choice([True, False])),
                (random.randint(1, width - 2), "6_stairs", random.randint(1, height - 4),
                 random.choice([-1, 1])),
                (random.randint(1, width - 2), "7_pipe", random.randint(2, height - 4))
            ]))
        return Individual_DE(g)

# Individual = Individual_Grid
Individual = Individual_DE

def tournament_selection(pop, k=4):
    # Pick k random individuals, return the best among them.
    candidates = random.sample(pop, k)
    candidates.sort(key=lambda ind: ind.fitness(), reverse=True)
    return candidates[0]

def generate_successors(population):
    # Generate a new population using tournament selection and crossover.
    results = []
    selected_parents = [tournament_selection(population) for _ in range(len(population) // 2)]
    additional_parents = [random.choice(population) for _ in range(len(population) // 2)]
    all_parents = selected_parents + additional_parents

    for i in range(0, len(all_parents) - 1, 2):
        parent1 = all_parents[i]
        parent2 = all_parents[i + 1]
        children = parent1.generate_children(parent2)  # ✅ Expecting 1 or 2 children

        # ✅ Make sure each child is added separately
        if isinstance(children, tuple):  # If two children are returned
            results.extend(children)
        else:  # If only one child is returned
            results.append(children)

    return results



def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("./levels/last.txt", 'w+') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1
                # STUDENT Determine stopping condition
                stop_condition = False
                if stop_condition:
                    break
                # STUDENT Also consider using FI-2POP as in the Sorenson & Pasquier paper
                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population


if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")