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
    This class uses a NeuroEvolution of Augmented Topologies (NEAT) algorithm
    to solve a maze.
    """
    def __init__(self, grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
        """
        Initialize the neatSolver object.

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

    
    def move(self, pos: list[int, int], action: int) -> list[int, int]:
        """
        Move the player by one step in a given direction.

        Parameters:
            pos (List[int, int]): Current position.
            action (int): Action to take.

        Returns:
            List[int, int]: New position after the move.
        """
        if action == UP:
                pos[ROW] -= 1
        elif action == DOWN:
                pos[ROW] += 1
        elif action == LEFT:
                pos[COL] -= 1
        elif action == RIGHT:
                pos[COL] += 1

        return pos
    

    def random_move(self, pos: list[int, int]) -> tuple[list[int, int], int]:
        """
        Move the player randomly among possible actions.

        Parameters:
            pos (List[int, int]): Current position.

        Returns:
            Tuple[List[int, int], int]: New position after the move and the action taken.
        """
        random_move = np.random.choice(self.possible_actions(self.grid, pos))
        pos = self.move(pos, random_move)

        return pos, random_move
    

    def novelty_score(self, path: list[tuple[int, int]]) -> float:
        """
        Calculate the novelty score based on unique positions.

        Parameters:
            path (List[Tuple[int, int]]): List of positions visited.

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


    def possible_actions(self, grid: np.ndarray, pos: tuple[int, int]) -> list[int]:
        """
        Return possible moves given a grid and a position.

        Parameters:
            grid (np.ndarray): The maze grid.
            pos (Tuple[int, int]): Current position.

        Returns:
            List[int]: List of possible actions.
        """
        actions = []
        if pos[ROW] - 1 >= 0 and grid[pos[ROW] - 1][pos[COL]] != 1:
            actions.append(UP)
        if pos[ROW] + 1 < len(grid) and grid[pos[ROW] + 1][pos[COL]] != 1:
            actions.append(DOWN)
        if pos[COL] - 1 >= 0 and grid[pos[ROW]][pos[COL] - 1] != 1:
            actions.append(LEFT)
        if pos[COL] + 1 < len(grid[ROW]) and grid[pos[ROW]][pos[COL] + 1] != 1:
            actions.append(RIGHT)

        return actions


    def distance_to_goal(self, pos: tuple[int, int], goal: tuple[int, int]) -> int:
        """
        Calculate Manhattan distance from current position to goal.

        Parameters:
            pos (Tuple[int, int]): Current position.
            goal (Tuple[int, int]): Goal position.

        Returns:
            int: Manhattan distance to goal.
        """
        return abs(pos[ROW] - goal[ROW]) + abs(pos[COL] - goal[COL])


    def reset_pos(self) -> list[int, int]:
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


    def eval_genomes(self, genomes, config) -> None:
        """
        Evaluate the genomes and assign fitness.

        Parameters:
            genomes: List of genomes to evaluate.
            config: NEAT configuration.
        """
        for genome_id, genome in genomes:  # genome_id is used by the neat-python library  
            fitness = self.calc_fitness(genome, config)
            genome.fitness = fitness


    def run(self, config_file: str, num_generations: int):
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

    for i in range(10):  # increase for learning on more mazes
        start_time = time.time()

        maze = MazeMaker(4, 4, 0.5, 7)  # NOTE: when changing maze size, change input paramater accordingly in config file.
        grid = maze.return_maze()
        start = (maze.return_start_coor()[ROW], maze.return_start_coor()[COL])
        goal = (maze.return_goal_coor()[ROW], maze.return_goal_coor()[COL])

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
