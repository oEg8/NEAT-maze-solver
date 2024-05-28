import os
import time
import neat
from MazeMaker import MazeMaker
import numpy as np
from Visualiser import Visualiser
import matplotlib.pyplot as plt
import pickle


LOCAL_DIR = os.path.dirname(__file__)
ROW = 0
COL = 1


def save_model(network, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(network, f)
        f.close()


def load_model(filename: str):
    return pickle.load(open(filename, 'rb')) 


class neatSolver:
    """
    This class uses a NeuroEvolution of Augmented Topologies (NEAT) algorithm
    to solve a maze.
    """
    def __init__(self, grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
        self.grid = grid
        self.start = start
        self.goal = goal

        self.max_steps = 100
        self.step_cost = 1
        self.goal_reward = 100
        self.illegal_pen = 3
        self.novel_pen = 1
        self.illegal_mutation_rate = 0.1

        self.winner_directions = []
        np.random.seed(1)

    
    def move(self, pos: list[int, int], action: int) -> list[int, int]:
        """
        This method moves the player 1 step to the desired position
        """
        if action == 0:  # up
                pos[ROW] -= 1
        elif action == 1:  # down
                pos[ROW] += 1
        elif action == 2:  # left
                pos[COL] -= 1
        elif action == 3:  # right
                pos[COL] += 1

        return pos
    

    def random_move(self, pos: list[int, int]) -> tuple[list[int, int], int]:
        """
        This method moves the player. The direction is chosen randomply out of the possible actions for that position.
        """
        random_move = np.random.choice(self.possible_actions(self.grid, pos))
        pos = self.move(pos, random_move)

        return pos, random_move
    

    def novelty_score(self, path: list[tuple[int, int]]) -> float:
        """
        This method counts the amount of times the algorithms has been on a
        unique posistion and penalizes with the novelty penalty * that amount
        """
        novel_score = 0
        unique_positions = []
        for item in path:
            if item not in unique_positions:
                unique_positions.append(item)

        # counts the amount of times the algorithms has been on a posistion and
        # penalizes with the novelty penalty * that amount
        for i in range(len(unique_positions)):
            pen_multiplier = path.count(unique_positions[i])
            if pen_multiplier > 1:
                novel_score += pen_multiplier-1 * self.novel_pen

        return novel_score


    def possible_actions(self, grid: np.ndarray, pos: tuple[int, int]) -> list:
        """
        This method returns a list of possible moves the player can take given
        a grid and a position
        """
        actions = []
        if pos[ROW] - 1 >= 0 and grid[pos[ROW] - 1][pos[COL]] != 1:
            actions.append(0)
        if pos[ROW] + 1 < len(grid) and grid[pos[ROW] + 1][pos[COL]] != 1:
            actions.append(1)
        if pos[COL] - 1 >= 0 and grid[pos[ROW]][pos[COL] - 1] != 1:
            actions.append(2)
        if pos[COL] + 1 < len(grid[ROW]) and grid[pos[ROW]][pos[COL] + 1] != 1:
            actions.append(3)

        return actions


    def distance_to_goal(self, pos: tuple[int, int], goal: tuple[int, int]) -> int:
        return abs(pos[ROW] - goal[ROW]) + abs(pos[COL] - goal[COL])


    def reset_pos(self) -> list[int, int]:
        """
        This method resets the position of the player to the starting position
        so the next iteration can start.
        """
        return [start[ROW], start[COL]]


    def calc_fitness(self, genome, config) -> int:
        """
        This method calculates the fitness for the given genome. This numberf
        is to be maximised.

        The fitness score consists of:
            - Amount of steps taken -
            - Absolute distance from current posision to goal -
            - Novelty score -
            - A penalty for illegal steps --
            - A reward for reaching the goal +++
        """
        fitness = 0
        pos = self.reset_pos()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        actions = []
        path = [pos]

        for _ in range(self.max_steps):
            # define the state which will be used as input for the algorithm
            state = list(grid.flatten())
            state.extend([pos[ROW], pos[COL], goal[ROW], goal[COL],
                          self.distance_to_goal(pos, goal)])

            # generate output from input (state)
            output = net.activate(state)

            action = np.argmax(output)
            possible_actions = self.possible_actions(grid, pos)

            # checks if the action is possible and take it if so, otherwise
            # step is illegal so genome will be penalized
            if action in possible_actions:
                fitness -= self.step_cost
                pos = self.move(pos, action)
                actions.append(action)
            else:
                fitness -= self.illegal_pen
                # to introduce randomness the genome will take a random action a percentage of times
                # (illegal_mutation_rate) the genome wants to take a illegal action
                if np.random.choice([0, 1], p=[1-self.illegal_mutation_rate, self.illegal_mutation_rate]):
                    pos, random_move = self.random_move(pos)
                    actions.append(random_move)

            path.append(pos)

            # checks if the goal is reached and breaks out of the loop if so
            if pos[ROW] == goal[ROW] and pos[COL] == goal[COL]:
                fitness += self.goal_reward
                # saves the winning directions for viualisation
                self.winner_directions = actions
                break

        fitness -= self.distance_to_goal(pos, goal)
        fitness -= self.novelty_score(path)

        return fitness


    def eval_genomes(self, genomes, config):
        """
        This method evaluates the genomes and creates the genome objects for
        the algorithm to use.
        """
        for genome_id, genome in genomes:  # genome_id is used by the neat-python library  
            fitness = self.calc_fitness(genome, config)
            genome.fitness = fitness


    def run(self, config_file, num_evals):
        """
        This method runs the algorithm.
        """
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population
        p = neat.Population(config)

        # check if an earlier trained model exists and use that best genome as the initial training genome
        if os.path.exists(os.path.join(LOCAL_DIR, 'best_neat_solver.plk')):
            # Load the previously trained model
            winner_net = load_model(os.path.join(LOCAL_DIR,
                                                         'best_neat_solver.plk'))

            # Create a genome from the loaded model's structure
            winner_genome = neat.DefaultGenome(1)  # Set the ID to 1
            winner_genome.configure_new(config.genome_config)
            winner_genome.nodes = winner_net.nodes
            winner_genome.connections = winner_net.connections
            winner_genome.fitness = winner_net.fitness

            p.population[0] = winner_genome


        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        p.run(self.eval_genomes, num_evals)

        # Save the best genome
        best_genome = stats.best_genome()
        save_model(best_genome, os.path.join(LOCAL_DIR,
                                                     'best_neat_solver.plk'))

        max_gen_fitness = best_genome.fitness
        num_generations = len(stats.get_fitness_mean())

        print('\nBest genome:\n{!s}'.format(best_genome))

        return max_gen_fitness, num_generations, self.winner_directions


if __name__ == '__main__':

    config_path = os.path.join(LOCAL_DIR, 'neat-configuration.txt')

    results = {"ID": [],
               "MAX_FITNESS": [],
               "NUM_GENERATIONS": [],
               "TIME_TO_SOLVE": []}

    for i in range(10):  # increase for learning on multiple mazes
        start_time = time.time()

        maze = MazeMaker(4, 4, 0.5, 7)
        grid = maze.return_maze()
        start = (maze.return_start_coor()[ROW], maze.return_start_coor()[COL])
        goal = (maze.return_goal_coor()[ROW], maze.return_goal_coor()[COL])

        if os.path.exists(os.path.join(LOCAL_DIR, 'grids')):
            np.save(os.path.join(LOCAL_DIR, f'grids/grid{i}.npy'), grid)
        else:
            os.mkdir(os.path.join(LOCAL_DIR, 'grids'))
            np.save(os.path.join(LOCAL_DIR, f'grids/grid{i}.npy'), grid)

        n = neatSolver(grid, start, goal)

        max_gen_fitness, num_generations, winner_directions = n.run(config_path, 300)

        end_time = time.time()
        time_to_solve = round(end_time - start_time, 2)

        results["ID"].append(i)
        results["MAX_FITNESS"].append(max_gen_fitness)
        results["NUM_GENERATIONS"].append(num_generations)
        results["TIME_TO_SOLVE"].append(time_to_solve)

    # print(results)

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 5))
    # ax1.plot(results['ID'], results['MAX_FITNESS'], label='MAX_FITNESS', color='blue')
    # ax1.set_xlabel('ID')
    # ax1.set_ylabel('MAX_FITNESS')
    # ax2.plot(results['ID'], results['NUM_GENERATIONS'], label='NUM_GENERATIONS', color='red')
    # ax2.set_xlabel('ID')
    # ax2.set_ylabel('NUM_GENERATIONS')
    # ax3.plot(results['ID'], results['TIME_TO_SOLVE'], label='TIME_TO_SOLVE', color='green')
    # ax3.set_xlabel('ID')
    # ax3.set_ylabel('TIME_TO_SOLVE')
    # plt.tight_layout()
    # plt.show()

    # print(winner_directions)
    Visualiser(grid, [start[ROW], start[COL]], goal, winner_directions).draw_maze()
