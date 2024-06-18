import os
import time
import neat
from MazeMaker import MazeMaker
import numpy as np
from Visualiser import Visualiser
import pickle

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


def save_model(network, filename: str):
    """Saves NEAT model with pickle"""
    with open(filename, 'wb') as f:
        pickle.dump(network, f)
        f.close()

def load_model(filename: str):
    """Loads NEAT model with pickle"""
    return pickle.load(open(filename, 'rb')) 


class neatSolver:
    """
    This class uses a NeuroEvolution of Augmented Topologies (NEAT) algorithm to solve a maze.


    Attributes
    __________
    move                : list[int, int]
                        Move the player by one step in a given direction.
    random_move         : tuple[list[int, int], int]
                        Move the player randomly among possible actions.
    novelty_score       : float
                        Calculate the novelty score based on unique positions.
    possible_actions    :list[int, int]
                        Return possible moves given a grid and a position.
    distance_to_goal    : int
                        Calculate Manhattan distance from current position to goal.
    reset_position      : list[int, int]
                        Reset the player position to the starting position.
    calc_fitness        : int
                        Calculate the fitness for a given genome.
    eval_genomes        : None
                        Evaluate the genomes and assign fitness.
    run                 : tuple[int, int, int]
                        Run the NEAT algorithm.
    """
    def __init__(self, grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
        """
        Initializes the neatSolver object.

        Parameters:
            grid (np.ndarray): The maze grid.
            start (Tuple[int, int]): The starting position.
            goal (Tuple[int, int]): The goal position.
        """
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

    
    def move(self, posistion: list[int, int], action: int) -> list[int, int]:
        """
        Move the player by one step in a given direction.

        Parameters:
            posistion (list): Current position.
            action (int): Action to take.

        Returns:
            list: New position after the move.
        """
        if action == UP:
                posistion[ROW] -= 1
        elif action == DOWN:
                posistion[ROW] += 1
        elif action == LEFT:
                posistion[COL] -= 1
        elif action == RIGHT:
                posistion[COL] += 1

        return posistion
    

    def random_move(self, position: list[int, int]) -> tuple[list[int, int], int]:
        """
        Move the player randomly among possible actions.

        Parameters:
            position (List[int, int]): Current position.

        Returns:
            Tuple[List[int, int], int]: New position after the move and the action taken.
        """
        random_move = np.random.choice(self.possible_actions(self.grid, position))
        position = self.move(position, random_move)

        return position, random_move
    

    def novelty_score(self, path: list[tuple[int, int]]) -> float:
        """
        Calculate the novelty score based on unique positions.

        Parameters:
            path (list): List of positions visited.

        Returns:
            float: Novelty score.
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


    def possible_actions(self, grid: np.ndarray, position: tuple[int, int]) -> list[int]:
        """
        Return possible moves given a grid and a position.

        Parameters:
            grid (np.ndarray): The maze grid.
            pos (tuple): Current position.

        Returns:
            list: List of possible actions.
        """
        actions = []
        if position[ROW] - 1 >= 0 and grid[position[ROW] - 1][position[COL]] != 1:
            actions.append(UP)
        if position[ROW] + 1 < len(grid) and grid[position[ROW] + 1][position[COL]] != 1:
            actions.append(DOWN)
        if position[COL] - 1 >= 0 and grid[position[ROW]][position[COL] - 1] != 1:
            actions.append(LEFT)
        if position[COL] + 1 < len(grid[ROW]) and grid[position[ROW]][position[COL] + 1] != 1:
            actions.append(RIGHT)

        return actions


    def distance_to_goal(self, position: tuple[int, int], goal: tuple[int, int]) -> int:
        """
        Calculate Manhattan distance from current position to goal.

        Parameters:
            position (tuple): Current position.
            goal (tuple): Goal position.

        Returns:
            int: Manhattan distance to goal.
        """
        return abs(position[ROW] - goal[ROW]) + abs(position[COL] - goal[COL])


    def reset_position(self) -> list[int, int]:
        """
        Reset the player position to the starting position.

        Returns:
            List[int, int]: Starting position.
        """
        return [start[ROW], start[COL]]


    def calc_fitness(self, genome, config: str) -> int:
        """
        Calculate the fitness for a given genome.

        Parameters:
            genome: Genome to calculate fitness for.
            config (str): NEAT configuration file path.

        Returns:
            int: Fitness value.
        """
        fitness = 0
        position = self.reset_position()
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        actions = []
        path = [position]

        for _ in range(self.max_steps):
            # define the state which will be used as input for the algorithm
            state = list(grid.flatten())
            state.extend([position[ROW], position[COL], goal[ROW], goal[COL],
                          self.distance_to_goal(position, goal)])

            # generate output from input (state)
            output = net.activate(state)

            action = np.argmax(output)
            possible_actions = self.possible_actions(grid, position)

            # checks if the action is possible and take it if so, otherwise
            # step is illegal so genome will be penalized
            if action in possible_actions:
                fitness -= self.step_cost
                position = self.move(position, action)
                actions.append(action)
            else:
                fitness -= self.illegal_pen
                # to introduce randomness the genome will take a random action a percentage of times
                # (illegal_mutation_rate) the genome wants to take a illegal action
                if np.random.choice([0, 1], p=[1-self.illegal_mutation_rate, self.illegal_mutation_rate]):
                    position, random_move = self.random_move(position)
                    actions.append(random_move)

            path.append(position)

            # checks if the goal is reached and breaks out of the loop if so
            if position[ROW] == goal[ROW] and position[COL] == goal[COL]:
                fitness += self.goal_reward
                # saves the winning directions for viualisation
                self.winner_directions = actions
                break

        fitness -= self.distance_to_goal(position, goal)
        fitness -= self.novelty_score(path)

        return fitness


    def eval_genomes(self, genomes, config: str) -> None:
        """
        Evaluate the genomes and assign fitness.

        Parameters:
            genomes: List of genomes to evaluate.
            config (str): NEAT configuration file path.
        """
        for genome_id, genome in genomes:  # genome_id is used by the neat-python library  
            fitness = self.calc_fitness(genome, config)
            genome.fitness = fitness


    def run(self, config_file: str, num_generations: int) -> tuple[int, int, int]:
        """
        Run the NEAT algorithm.

        Parameters:
            config_file (str): Path to NEAT configuration file.
            num_generations (int): Maximum nomber of generations allowed.
        
        Returns:
            tuple[int, int, int]:
                - max_gen_fitness: Maximum generation fitness.
                - num_generations: Number of generations completed.
                - winner_directions: Directions of best genome
        """
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population
        p = neat.Population(config)

        # check if an earlier trained model exists and use that best genome as the initial training genome
        if os.path.exists('best_neat_solver.plk'):
            # Load the previously trained model
            winner_net = load_model('best_neat_solver.plk')

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

        p.run(self.eval_genomes, num_generations)

        # Save the best genome
        best_genome = stats.best_genome()
        save_model(best_genome, 'best_neat_solver.plk')

        max_gen_fitness = best_genome.fitness
        num_generations = len(stats.get_fitness_mean())

        print('\nBest genome:\n{!s}'.format(best_genome))

        return max_gen_fitness, num_generations, self.winner_directions


if __name__ == '__main__':

    config_path = 'neat-configuration.txt'

    results = {"ID": [],
               "MAX_FITNESS": [],
               "NUM_GENERATIONS": [],
               "TIME_TO_SOLVE": []}

    for i in range(2):  # increase for learning on more mazes
        start_time = time.time()

        maze = MazeMaker(4, 4, 0.5, 7)  # NOTE: when changing maze size, change input paramater accordingly in config file.
        grid = maze.get_maze()
        start = (maze.get_start()[ROW], maze.get_start()[COL])
        goal = (maze.get_goal()[ROW], maze.get_goal()[COL])

        if os.path.exists('grids'):
            np.save(f'grids/grid{i}.npy', grid)
        else:
            os.mkdir('grids')
            np.save(f'grids/grid{i}.npy', grid)

        n = neatSolver(grid, start, goal)

        max_gen_fitness, num_generations, winner_directions = n.run(config_path, 300)

        end_time = time.time()
        time_to_solve = round(end_time - start_time, 2)

        results["ID"].append(i)
        results["MAX_FITNESS"].append(max_gen_fitness)
        results["NUM_GENERATIONS"].append(num_generations)
        results["TIME_TO_SOLVE"].append(time_to_solve)


    Visualiser(grid, [start[ROW], start[COL]], goal, winner_directions).draw_maze()
