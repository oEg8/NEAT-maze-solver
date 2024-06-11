import numpy as np

ROW = 0
COL = 1

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


class MazeMaker:
    """
    This MazeMaker class generates a randomized maze containing 4 values.
    0: empty cell (the path)
    1: wall
    2: start
    3: end
    """
    def __init__(self, rows: int, columns: int, obstacle_ratio: float, minimum_route_length: int, seed: int = None) -> None:
        """
        Initialize the MazeMaker object.

        Parameters:
            rows (int): The number of rows in the maze.
            columns (int): The number of columns in the maze.
            obstacle_ratio (float): The ratio of obstacles to empty cells.
            minimum_route_length (int): The minimum required length of the route.
        """
        self.rows = rows
        self.columns = columns
        self.obstacle_ratio = obstacle_ratio
        self.minimum_route_length = minimum_route_length
        self.seed = np.random.seed(seed)

        self.final_grid = self.build_grid(self.obstacle_ratio)


    def random_start_goal(self) -> tuple:
        """
        Generates random start coordinate on the maze's border and sets goal coordinate opposite from start.

        Returns:
            tuple: Tuple containing start coordinates and goal coordinates.
        """
        rng_side = np.random.randint(0, 4)

        if rng_side == UP:
            rng_start = (0, np.random.randint(0, self.columns))
            rng_goal = (self.rows-1, self.columns-rng_start[COL]-1)
        elif rng_side == DOWN:
            rng_start = (self.rows-1, np.random.randint(0, self.columns))
            rng_goal = (0, self.columns-rng_start[COL]-1)
        elif rng_side == LEFT:
            rng_start = (np.random.randint(0, self.rows), 0)
            rng_goal = (self.rows-rng_start[ROW]-1, self.columns-1)
        elif rng_side == RIGHT:
            rng_start = (np.random.randint(0, self.rows), self.columns-1)
            rng_goal = (self.rows-rng_start[ROW]-1, 0)

        return rng_start, rng_goal


    def get_neighbours(self, square: tuple, search_grid: np.ndarray) -> list:
        """
        Get valid neighbours for a given square.

        Parameters:
            square (tuple): The coordinates of the square.
            search_grid (np.ndarray): The grid to search for neighbours.

        Returns:
            list: A list of valid neighbours.
        """
        neighbours = []

        if square[ROW] < (self.rows -1) and search_grid[square[ROW]+1][square[COL]] < -1:
            neighbours.append((square[ROW]+1, square[COL]))
        if square[ROW] > 0 and search_grid[square[ROW]-1][square[COL]] < -1:
            neighbours.append((square[ROW]-1, square[COL]))
        if square[COL] > 0 and search_grid[square[ROW]][square[COL]-1] < -1:
            neighbours.append((square[ROW], square[COL]-1))           
        if square[COL] < (self.columns -1) and search_grid[square[ROW]][square[COL]+1] < -1:
            neighbours.append((square[ROW], square[COL]+1))

        return neighbours


    def get_neighbours_with_values(self, square: tuple, search_grid: np.ndarray, value: int) -> tuple:
        """
        Get neighbouring squares with a specific value.

        Parameters:
            square (tuple): The coordinates of the square.
            search_grid (np.ndarray): The grid to search for neighbours.
            value (int): The value to search for.

        Returns:
            tuple: The coordinates of the neighbouring square with the specified value.
        """
        if square[ROW]+1 < self.rows and search_grid[square[ROW]+1][square[COL]] == value:
            return (square[ROW]+1, square[COL])
        elif square[ROW]-1 >= 0 and search_grid[square[ROW]-1][square[COL]] == value:
            return (square[ROW]-1, square[COL])
        elif square[COL]-1 >= 0 and search_grid[square[ROW]][square[COL]-1] == value:
            return (square[ROW], square[COL]-1)
        elif square[COL]+1 < self.columns and search_grid[square[ROW]][square[COL]+1] == value:
            return (square[ROW], square[COL]+1)


    def find_route(self, start_coor: tuple, end_coor: tuple, grid_with_obstacles: np.ndarray) -> list:
        """
        Find a route using breadth-first search.

        Parameters:
            start_coor (tuple): The coordinates of the starting point.
            end_coor (tuple): The coordinates of the ending point.
            grid_with_obstacles (np.ndarray): The grid with obstacles.

        Returns:
            list: A list of coordinates representing the route.
        """
        route = []
        search_grid = np.full((self.rows, self.columns), -3, dtype=np.int32)
        search_grid[(start_coor[ROW], start_coor[COL])] = 0
        search_grid[(end_coor[ROW], end_coor[COL])] = -2
        for i in range(self.rows):
            for j in range(self.columns):
                if grid_with_obstacles[i][j] == 1:
                    search_grid[i][j] = -1

        length = 0
        new_expand = [(start_coor[ROW], start_coor[COL])]
        searching = True
        while searching:
            to_expand = []
            to_expand.extend(new_expand)
            new_expand = []
            for square in to_expand:
                neighbours = self.get_neighbours(square, search_grid)
                for n in neighbours:
                    if search_grid[n[ROW]][n[COL]] == -2: # exit found
                        searching = False
                        search_grid[n[ROW]][n[COL]] = length+1 # increase length of path
                    elif search_grid[n[ROW]][n[COL]] == -3: # path
                        search_grid[n[ROW]][n[COL]] = length+1 # increase length of path
                        new_expand.append((n[ROW], n[COL]))
            length += 1
            if len(new_expand) == 0:
                searching = False

        if search_grid[end_coor[ROW]][end_coor[COL]] > 0:
            max_length = search_grid[end_coor[ROW]][end_coor[COL]]
            route.append((end_coor[ROW], end_coor[COL]))
            square = (end_coor[ROW], end_coor[COL])
            while max_length > 0:
                next_square = self.get_neighbours_with_values(square,
                                                              search_grid,
                                                              max_length-1)
                route.append(next_square)
                square = next_square
                max_length -= 1
            route.reverse()
        return route


    def path_check(self, min_route_length: int) -> bool:
        """
        Check if a route exists and meets the minimum length requirement.

        Parameters:
            min_route_length (int): The minimum length of the route.

        Returns:
            bool: True if a valid route exists, False otherwise.
        """
        found_route = self.find_route(self.start_coor,
                                      self.goal_coor,
                                      self.final_grid)
        if len(found_route) >= min_route_length:
            self.optimal_route = found_route
            return True
        else:
            self.optimal_route = None
            return False


    def build_grid(self, obstacle_ratio: float) -> np.ndarray:
        """
        Creates obstacles for the maze based on the obstacle ratio.

        Returns:
            np.ndarray: Grid with obstacles.
        """
        while True:
            self.final_grid = np.random.choice([0, 1],
                                                        (self.rows, self.columns),
                                                        p=[1-obstacle_ratio, obstacle_ratio])  # randomly creates obstacles
            self.start_coor, self.goal_coor = self.random_start_goal()
            self.final_grid[self.start_coor] = 2
            self.final_grid[self.goal_coor] = 3
            path_exists = self.path_check(self.minimum_route_length)
            if path_exists:
                break

        return self.final_grid


    def return_maze(self) -> np.ndarray:
        """
        Returns the generated maze with obstacles.

        Returns:
            np.ndarray: The maze grid.
        """
        return self.final_grid


    def return_optimal_route(self) -> list:
        """
        Returns the optimal route found in the maze.

        Returns:
            list: List of tuples representing the optimal route.
        """
        return self.optimal_route


    def return_directions(self, step_list: list) -> list[int]:
        """
        Return directions taken given a path.

        Parameters:
            step_list (list): The list of steps representing the path.

        Returns:
            list: A list of directions.
        """
        directions = []
        for i in range(len(step_list) - 1):
            current_step = step_list[i]
            next_step = step_list[i + 1]
            direction = (next_step[ROW] - current_step[ROW], 
                         next_step[COL] - current_step[COL])

            # Maps direction to a numeric value
            if direction == (-1, 0):
                directions.append(UP)  # up
            elif direction == (1, 0):
                directions.append(DOWN)  # down
            elif direction == (0, -1):
                directions.append(LEFT)  # left
            elif direction == (0, 1):
                directions.append(RIGHT)  # right

        return directions


    def return_start_coor(self) -> tuple[int]:
        """
        Returns the start coordinates.

        Returns:
            tuple (int): Start coordinates.
        """
        return self.start_coor


    def return_goal_coor(self) -> tuple[int]:
        """
        Returns the goal coordinates.

        Returns:
            tuple (int): Goal coordinates.
        """
        return self.goal_coor


if __name__ == '__main__':
    maze = MazeMaker(10, 10, 0.5, 10)
    print(maze.return_maze())