import pygame
import sys
import math
import random
import time
from itertools import permutations

# Initialize pygame
pygame.init()

# Constants
TITLE = "TSP-Sokoban"
TILE_SIZE = 40
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
DARK_GRAY = (100, 100, 100)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)
LIGHT_BLUE = (173, 216, 230)

# Game elements
WALL = '#'
PLAYER = '@'
TARGET = '.'
FLOOR = ' '
BOX = '$'

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()

# Fonts
font = pygame.font.SysFont('Arial', 18)
small_font = pygame.font.SysFont('Arial', 14)
large_font = pygame.font.SysFont('Arial', 24)

class TSPSolver:
    def __init__(self, start_pos, targets, grid, max_iterations=100):
        self.start_pos = start_pos
        self.targets = targets[:]
        self.grid = grid
        self.max_iterations = max_iterations
        self.best_path = None
        self.best_distance = float('inf')
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def calculate_path_distance(self, path):
        """Calculate total distance for a path including obstacles"""
        total_distance = 0
        current_pos = self.start_pos
        
        for target in path:
            # Use A* to find actual path distance considering obstacles
            distance = self.astar_distance(current_pos, target)
            if distance == float('inf'):
                return float('inf')  # Path not possible
            total_distance += distance
            current_pos = target
        
        return total_distance
    
    def astar_distance(self, start, goal):
        """A* algorithm to find shortest path distance considering obstacles"""
        if start == goal:
            return 0
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        visited = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while open_set:
            current_f, current = min(open_set)
            open_set.remove((current_f, current))
            
            if current == goal:
                return g_score[current]
            
            visited.add(current)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds and walls
                if (neighbor[1] < 0 or neighbor[1] >= len(self.grid) or 
                    neighbor[0] < 0 or neighbor[0] >= len(self.grid[0]) or
                    self.grid[neighbor[1]][neighbor[0]] == WALL or
                    neighbor in visited):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, goal)
                    
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))
        
        return float('inf')  # No path found
    
    def solve_with_simulated_annealing(self):
        """Solve TSP using simulated annealing"""
        if not self.targets:
            return []
        
        # Initialize with random permutation
        current_path = self.targets[:]
        random.shuffle(current_path)
        current_distance = self.calculate_path_distance(current_path)
        
        best_path = current_path[:]
        best_distance = current_distance
        
        temperature = 1000.0
        cooling_rate = 0.995
        
        print(f"Starting TSP solver with {len(self.targets)} targets...")
        
        for iteration in range(self.max_iterations):
            # Generate neighbor by swapping two random cities
            new_path = current_path[:]
            if len(new_path) > 1:
                i, j = random.sample(range(len(new_path)), 2)
                new_path[i], new_path[j] = new_path[j], new_path[i]
            
            new_distance = self.calculate_path_distance(new_path)
            
            # Accept or reject the new solution
            if new_distance < current_distance:
                current_path = new_path
                current_distance = new_distance
                
                if new_distance < best_distance:
                    best_path = new_path[:]
                    best_distance = new_distance
            else:
                # Accept worse solution with probability
                if temperature > 0:
                    probability = math.exp(-(new_distance - current_distance) / temperature)
                    if random.random() < probability:
                        current_path = new_path
                        current_distance = new_distance
            
            temperature *= cooling_rate
            
            if iteration % 200 == 0:
                print(f"Iteration {iteration}: Best distance = {best_distance:.1f}")
        
        self.best_path = best_path
        self.best_distance = best_distance
        print(f"TSP solved! Best distance: {best_distance:.1f}")
        return best_path

class PathFinder:
    """A* pathfinder for navigation between points"""
    def __init__(self, grid):
        self.grid = grid
    
    def find_path(self, start, goal):
        """Find path from start to goal using A*"""
        if start == goal:
            return [start]
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.manhattan_distance(start, goal)}
        visited = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while open_set:
            current_f, current = min(open_set)
            open_set.remove((current_f, current))
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            visited.add(current)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor[1] < 0 or neighbor[1] >= len(self.grid) or 
                    neighbor[0] < 0 or neighbor[0] >= len(self.grid[0]) or
                    self.grid[neighbor[1]][neighbor[0]] == WALL or
                    neighbor in visited):
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.manhattan_distance(neighbor, goal)
                    
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class TSPSokobanGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("TSP Sokoban")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        self.current_level = 0
        self.levels = self.create_levels()
        self.load_level(self.current_level)
        self.auto_solve = False
        self.auto_solve_path = []
        self.auto_solve_index = 0
        self.auto_solve_delay = 0.2
        self.last_auto_move_time = 0
        self.show_edit_mode = False
        self.edit_tool = WALL
        self.show_solution = False
        self.solution_path = []
        self.moves_count = 0
        self.best_ai_distance = float('inf')
        self.player_distance = 0
        self.game_completed = False
        self.show_stats = False
        self.level_complete = False
        self.race_mode = False
        self.ai_progress = 0
        self.last_ai_move = time.time()
        self.ai_move_interval = 1.0  # AI moves every 1 second

    def create_levels(self):
        """Load levels from levels.txt file"""
        levels = []
        current_level = []
        
        try:
            with open('combinedGame/levels.txt', 'r') as file:
                for line in file:
                    line = line.rstrip()
                    if line.startswith('//') or line == '':
                        if current_level:
                            levels.append(current_level)
                            current_level = []
                    else:
                        current_level.append(line)
                if current_level:
                    levels.append(current_level)
        except FileNotFoundError:
            print("Error: levels.txt not found!")
            # Add a simple default level
            levels.append([
                "########",
                "#@     #",
                "#  .   #",
                "#    . #",
                "########"
            ])
        
        return levels
    
    def load_level(self, level_index):
        """Load a specific level"""
        if 0 <= level_index < len(self.levels):
            level_data = self.levels[level_index]
            self.grid = []
            self.player_pos = [0, 0]
            self.targets = []
            self.visited_targets = set()
            
            for y, row in enumerate(level_data):
                grid_row = []
                for x, cell in enumerate(row):
                    if cell == PLAYER:
                        self.player_pos = [x, y]
                        grid_row.append(FLOOR)
                    elif cell == TARGET:
                        self.targets.append((x, y))
                        grid_row.append(FLOOR)
                    else:
                        grid_row.append(cell)
                self.grid.append(list(grid_row))
            
            self.current_level = level_index
            self.pathfinder = PathFinder(self.grid)
            self.auto_solve = False
            self.auto_solve_path = []
            self.show_solution = False
            self.solution_path = []
            self.moves_count = 0
            self.player_distance = 0
            self.game_completed = False
            self.show_stats = False
            self.best_ai_distance = float('inf')
            print(f"Loaded level {level_index + 1}")
    
    def is_valid_move(self, x, y):
        """Check if position is valid"""
        return (0 <= y < len(self.grid) and 
                0 <= x < len(self.grid[0]) and 
                self.grid[y][x] != WALL)
    
    def move_player(self, dx, dy):
        """Move player if possible"""
        new_x = self.player_pos[0] + dx
        new_y = self.player_pos[1] + dy
        
        if self.is_valid_move(new_x, new_y):
            self.player_pos = [new_x, new_y]
            
            # Check if player reached a target
            player_tuple = tuple(self.player_pos)
            if player_tuple in self.targets and player_tuple not in self.visited_targets:
                self.visited_targets.add(player_tuple)
                print(f"Reached target! ({len(self.visited_targets)}/{len(self.targets)})")
            
            moved = True
        else:
            moved = False
        
        if moved:
            self.moves_count += 1
            # Update player's total distance
            self.player_distance += 1
            self.player_moves += 1

            if self.is_level_complete():
                self.player_time = time.time() - self.start_time
                self.determine_winner()

        return moved
    
    def is_level_complete(self):
        """Check if all targets have been visited"""
        completed = len(self.visited_targets) == len(self.targets)
        
        if completed:
            self.game_completed = True
            self.show_stats = True
        return completed
    
    def solve_tsp(self):
        """Solve the TSP and generate movement path"""
        if not self.targets:
            return
        
        # Get unvisited targets
        unvisited = [t for t in self.targets if t not in self.visited_targets]
        if not unvisited:
            print("All targets already visited!")
            return
        
        print("Solving TSP...")
        solver = TSPSolver(tuple(self.player_pos), unvisited, self.grid)
        optimal_order = solver.solve_with_simulated_annealing()
        
        if not optimal_order:
            print("No solution found!")
            return
        
        # Generate actual movement path
        current_pos = tuple(self.player_pos)
        full_path = []
        
        for target in optimal_order:
            path_segment = self.pathfinder.find_path(current_pos, target)
            if path_segment:
                full_path.extend(path_segment[1:])  # Skip the first position (current)
                current_pos = target
        
        self.solution_path = optimal_order
        return full_path
    
    def toggle_auto_solve(self):
        """Toggle auto-solve mode"""
        if not self.auto_solve:
            path = self.solve_tsp()
            if path:
                self.auto_solve = True
                self.auto_solve_path = path
                self.auto_solve_index = 0
                print(f"Auto-solve started with {len(path)} moves")
            else:
                print("Could not generate solution path")
        else:
            self.auto_solve = False
            print("Auto-solve stopped")
    
    def auto_solve_step(self):
        """Execute one step of auto-solve"""
        if not self.auto_solve or not self.auto_solve_path:
            return
        
        current_time = time.time()
        if current_time - self.last_auto_move_time < self.auto_solve_delay:
            return
        
        if self.auto_solve_index < len(self.auto_solve_path):
            target_pos = self.auto_solve_path[self.auto_solve_index]
            dx = target_pos[0] - self.player_pos[0]
            dy = target_pos[1] - self.player_pos[1]
            
            if abs(dx) + abs(dy) == 1:  # Adjacent position
                self.move_player(dx, dy)
                self.auto_solve_index += 1
                self.last_auto_move_time = current_time
            else:
                # Skip to next position if not adjacent (shouldn't happen with good pathfinding)
                self.auto_solve_index += 1
        else:
            self.auto_solve = False
            print("Auto-solve completed!")
    
    def toggle_edit_mode(self):
        """Toggle map editing mode"""
        self.show_edit_mode = not self.show_edit_mode
        if self.show_edit_mode:
            print("Edit mode ON - Click to place/remove elements")
        else:
            print("Edit mode OFF")
    
    def handle_edit_click(self, pos):
        """Handle mouse click in edit mode"""
        if not self.show_edit_mode:
            return
        
        # Calculate grid position from screen coordinates
        board_width = len(self.grid[0]) * TILE_SIZE
        board_height = len(self.grid) * TILE_SIZE
        board_offset_x = 200  # UI panel width
        board_offset_y = (SCREEN_HEIGHT - board_height) // 2
        
        grid_x = (pos[0] - board_offset_x) // TILE_SIZE
        grid_y = (pos[1] - board_offset_y) // TILE_SIZE
        
        if (0 <= grid_x < len(self.grid[0]) and 0 <= grid_y < len(self.grid)):
            if self.edit_tool == PLAYER:
                # Move player
                if self.grid[grid_y][grid_x] == FLOOR:
                    self.player_pos = [grid_x, grid_y]
            elif self.edit_tool == TARGET:
                # Toggle target
                target_pos = (grid_x, grid_y)
                if target_pos in self.targets:
                    self.targets.remove(target_pos)
                    self.visited_targets.discard(target_pos)
                else:
                    self.targets.append(target_pos)
                    if self.grid[grid_y][grid_x] == WALL:
                        self.grid[grid_y][grid_x] = FLOOR
            else:
                # Place wall or floor
                if (grid_x, grid_y) != tuple(self.player_pos):
                    self.grid[grid_y][grid_x] = self.edit_tool
    
    def reset_level(self):
        self.load_level(self.levels[self.current_level])
        self.show_stats = False
        self.level_complete = False
        self.race_mode = False
        self.player_moves = 0
        self.ai_moves = 0
        self.start_time = None

    def next_level(self):
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self.load_level(self.levels[self.current_level])
            self.show_stats = False
            self.level_complete = False
            self.race_mode = False
            self.player_moves = 0
            self.ai_moves = 0
            self.start_time = None
        else:
            # Show final game completion screen
            self.show_game_complete = True

    def determine_winner(self):
        self.show_stats = True
        self.level_complete = True
        if self.compare_by_moves:
            self.player_won = self.player_moves < len(self.solution_path)
        else:
            ai_estimated_time = len(self.solution_path) * self.ai_move_interval
            self.player_won = self.player_time < ai_estimated_time

    def update_race(self):
        """Updates AI position during race mode"""
        if not self.race_mode:
            return

        current_time = time.time()
        
        # Move AI if enough time has passed
        if current_time - self.last_ai_move >= self.ai_move_interval:
            if self.ai_progress < len(self.solution_path):
                self.ai_current_pos = self.solution_path[self.ai_progress]
                self.ai_progress += 1
                self.last_ai_move = current_time
                self.ai_moves += 1

    def draw(self):
        # Clear screen with background color
        self.screen.fill((200, 200, 200))  # Light gray background

        # Draw tiles
        for y in range(len(self.current_map)):
            for x in range(len(self.current_map[y])):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                
                # Draw different tiles
                if self.current_map[y][x] == WALL:
                    pygame.draw.rect(self.screen, (100, 100, 100), rect)  # Gray walls
                elif self.current_map[y][x] == TARGET:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # White floor
                    pygame.draw.circle(self.screen, (0, 255, 0), 
                                    (x * TILE_SIZE + TILE_SIZE//2, 
                                     y * TILE_SIZE + TILE_SIZE//2), 
                                    TILE_SIZE//4)  # Green target
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)  # White floor

        # Draw player
        pygame.draw.circle(self.screen, (0, 0, 255),
                         (self.player_pos[0] * TILE_SIZE + TILE_SIZE//2,
                          self.player_pos[1] * TILE_SIZE + TILE_SIZE//2),
                         TILE_SIZE//3)  # Blue player

        # Update display
        pygame.display.flip()

def main():
    game = TSPSokobanGame()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if game.show_stats:
                    game.next_level()
                    continue
                    
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    game.move_player(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    game.move_player(1, 0)
                elif event.key == pygame.K_UP:
                    game.move_player(0, -1)
                elif event.key == pygame.K_DOWN:
                    game.move_player(0, 1)
                elif event.key == pygame.K_s:
                    game.toggle_auto_solve()
                elif event.key == pygame.K_r:
                    game.reset_level()
                elif event.key == pygame.K_n:
                    game.next_level()
                elif event.key == pygame.K_p:
                    game.prev_level()
                elif event.key == pygame.K_e:
                    game.toggle_edit_mode()
                elif event.key == pygame.K_t:
                    game.show_solution = not game.show_solution
                    if game.show_solution and not game.solution_path:
                        game.solve_tsp()
                elif event.key == pygame.K_1:
                    game.edit_tool = WALL
                elif event.key == pygame.K_2:
                    game.edit_tool = FLOOR
                elif event.key == pygame.K_3:
                    game.edit_tool = TARGET
                elif event.key == pygame.K_4:
                    game.edit_tool = PLAYER
                elif event.key == pygame.K_SPACE:
                    if not game.race_mode:
                        game.start_race()
                elif event.key == pygame.K_h:
                    game.show_race_ui = not game.show_race_ui
            elif event.type == pygame.MOUSEBUTTONDOWN:
                game.handle_edit_click(pygame.mouse.get_pos())
        
        game.draw()
        game.clock.tick(60)  # Limit to 60 FPS

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()