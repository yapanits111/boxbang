import pygame
import sys
import math
import random
from datetime import datetime, timedelta
import os
import time
import copy

# Initialize pygame
pygame.init()

# Constants
TITLE = "BoxBang"
TILE_SIZE = 50
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
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

# Game elements
WALL = '#'
PLAYER = '@'
CRATE = '$'
TARGET = '.'
CRATE_ON_TARGET = '*'
PLAYER_ON_TARGET = '+'
FLOOR = ' '

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()

# Font
font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 16)

class GameState:
    def __init__(self, level_grid, player_pos, crates_pos, targets_pos):
        self.grid = level_grid
        self.player_pos = player_pos[:]
        self.crates_pos = [crate[:] for crate in crates_pos]
        self.targets_pos = targets_pos[:]
        self.move_count = 0
        self.moves_history = []
    
    def copy(self):
        return GameState(
            [row[:] for row in self.grid],
            self.player_pos[:],
            [crate[:] for crate in self.crates_pos],
            self.targets_pos[:]
        )
    
    def evaluate(self):
        """Evaluation function for simulated annealing"""
        if self.is_solved():
            return 0
        
        total_distance = 0
        
        # Calculate minimum total distance from crates to targets
        for crate in self.crates_pos:
            min_dist = float('inf')
            for target in self.targets_pos:
                dist = abs(crate[0] - target[0]) + abs(crate[1] - target[1])
                min_dist = min(min_dist, dist)
            total_distance += min_dist
        
        # Add penalty for deadlocks
        deadlock_penalty = 0
        for crate in self.crates_pos:
            if self._is_deadlock(crate):
                deadlock_penalty += 100
        
        # Add penalty for crates blocking each other
        blocking_penalty = 0
        for i, crate1 in enumerate(self.crates_pos):
            for j, crate2 in enumerate(self.crates_pos[i+1:], i+1):
                if abs(crate1[0] - crate2[0]) + abs(crate1[1] - crate2[1]) == 1:
                    blocking_penalty += 5
        
        return total_distance + deadlock_penalty + blocking_penalty

    def _is_deadlock(self, crate):
        """Check if a crate is in a deadlock position"""
        x, y = crate
        
        # If on target, not deadlocked
        if [x, y] in self.targets_pos:
            return False
        
        # Check for corner deadlock
        left_blocked = x == 0 or self.grid[y][x-1] == WALL
        right_blocked = x == len(self.grid[0])-1 or self.grid[y][x+1] == WALL
        top_blocked = y == 0 or self.grid[y-1][x] == WALL
        bottom_blocked = y == len(self.grid)-1 or self.grid[y+1][x] == WALL
        
        # Corner deadlock
        if (left_blocked or right_blocked) and (top_blocked or bottom_blocked):
            return True
        
        return False
    
    def is_solved(self):
        """Check if all crates are on targets"""
        crates_on_targets = 0
        for crate in self.crates_pos:
            if crate in self.targets_pos:
                crates_on_targets += 1
        return crates_on_targets == len(self.targets_pos)
    
    def get_possible_moves(self):
        """Get all possible moves from current state"""
        moves = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        
        for dx, dy in directions:
            new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
            
            # Check bounds
            if (new_y < 0 or new_y >= len(self.grid) or 
                new_x < 0 or new_x >= len(self.grid[0])):
                continue
            
            # Check wall
            if self.grid[new_y][new_x] == WALL:
                continue
            
            # Check if there's a crate
            crate_at_pos = None
            for crate in self.crates_pos:
                if crate[0] == new_x and crate[1] == new_y:
                    crate_at_pos = crate
                    break
            
            if crate_at_pos:
                # Check if crate can be pushed
                new_crate_x, new_crate_y = new_x + dx, new_y + dy
                
                # Check bounds for crate
                if (new_crate_y < 0 or new_crate_y >= len(self.grid) or 
                    new_crate_x < 0 or new_crate_x >= len(self.grid[0])):
                    continue
                
                # Check wall for crate
                if self.grid[new_crate_y][new_crate_x] == WALL:
                    continue
                
                # Check if another crate is there
                crate_blocked = False
                for other_crate in self.crates_pos:
                    if other_crate[0] == new_crate_x and other_crate[1] == new_crate_y:
                        crate_blocked = True
                        break
                
                if not crate_blocked:
                    moves.append((dx, dy))
            else:
                moves.append((dx, dy))
        
        return moves
    
    def apply_move(self, dx, dy):
        """Apply a move and return new state"""
        new_state = self.copy()
        new_x, new_y = new_state.player_pos[0] + dx, new_state.player_pos[1] + dy
        
        # Check if there's a crate to push
        for i, crate in enumerate(new_state.crates_pos):
            if crate[0] == new_x and crate[1] == new_y:
                # Push the crate
                new_state.crates_pos[i][0] += dx
                new_state.crates_pos[i][1] += dy
                break
        
        # Move the player
        new_state.player_pos[0] = new_x
        new_state.player_pos[1] = new_y
        new_state.move_count += 1
        
        return new_state

class SimulatedAnnealingSolver:
    def __init__(self, initial_state, max_iterations=1000, initial_temp=100, cooling_rate=0.95):
        self.initial_state = initial_state
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.solution_path = []
    
    def solve(self):
        """Solve using simulated annealing"""
        current_state = self.initial_state.copy()
        current_cost = current_state.evaluate()
        
        best_state = current_state.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        path = []
        
        print(f"Starting solver with initial cost: {current_cost}")
        
        for iteration in range(self.max_iterations):
            if current_state.is_solved():
                print(f"Solution found in {iteration} iterations!")
                self.solution_path = path
                return path
            
            # Get possible moves
            possible_moves = current_state.get_possible_moves()
            if not possible_moves:
                break
            
            # Choose a random move
            move = random.choice(possible_moves)
            new_state = current_state.apply_move(move[0], move[1])
            new_cost = new_state.evaluate()
            
            # Calculate acceptance probability
            if new_cost < current_cost:
                # Better solution, accept it
                current_state = new_state
                current_cost = new_cost
                path.append(move)
                
                if new_cost < best_cost:
                    best_state = new_state.copy()
                    best_cost = new_cost
            else:
                # Worse solution, accept with probability
                if temperature > 0:
                    probability = math.exp(-(new_cost - current_cost) / temperature)
                    if random.random() < probability:
                        current_state = new_state
                        current_cost = new_cost
                        path.append(move)
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Print progress occasionally
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Cost: {current_cost}, Temp: {temperature:.2f}")
        
        print(f"Solver finished. Best cost: {best_cost}")
        return None

class BoxBangGame:
    def __init__(self):
        self.level_num = 1
        self.auto_solve = False
        self.auto_solve_delay = 0.5  # seconds between auto moves
        self.last_auto_move_time = 0
        self.solver = None
        self.solution_moves = []
        self.current_move_index = 0
        self.load_level(self.level_num)
    
    def load_level(self, level_num):
        """Load a level from file or use a default level"""

        with open(f"lvl{level_num}.txt", "r") as f:
            level_data = f.read().strip().split('\n')
        
        self.grid = []
        self.player_pos = [0, 0]
        self.crates_pos = []
        self.targets_pos = []
        
        for y, row in enumerate(level_data):
            grid_row = []
            for x, cell in enumerate(row):
                if cell == PLAYER:
                    self.player_pos = [x, y]
                    grid_row.append(FLOOR)
                elif cell == CRATE:
                    self.crates_pos.append([x, y])
                    grid_row.append(FLOOR)
                elif cell == TARGET:
                    self.targets_pos.append([x, y])
                    grid_row.append(FLOOR)
                elif cell == CRATE_ON_TARGET:
                    self.crates_pos.append([x, y])
                    self.targets_pos.append([x, y])
                    grid_row.append(FLOOR)
                elif cell == PLAYER_ON_TARGET:
                    self.player_pos = [x, y]
                    self.targets_pos.append([x, y])
                    grid_row.append(FLOOR)
                else:
                    grid_row.append(cell)
            self.grid.append(grid_row)
        
        self.move_count = 0
        self.moves_history = []
        self.game_state = GameState(self.grid, self.player_pos, self.crates_pos, self.targets_pos)
        
        # Reset auto-solve state
        self.auto_solve = False
        self.solver = None
        self.solution_moves = []
        self.current_move_index = 0
    
    def get_current_state(self):
        """Get current game state"""
        return GameState(
            [row[:] for row in self.grid],
            self.player_pos[:],
            [crate[:] for crate in self.crates_pos],
            self.targets_pos[:]
        )
    
    def is_valid_move(self, dx, dy):
        """Check if a move is valid"""
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
        
        # Check if position is within grid bounds
        if (new_y < 0 or new_y >= len(self.grid) or 
            new_x < 0 or new_x >= len(self.grid[0])):
            return False
        
        # Check if position has a wall
        if self.grid[new_y][new_x] == WALL:
            return False
        
        # Check if position has a crate
        crate_idx = -1
        for i, crate in enumerate(self.crates_pos):
            if crate[0] == new_x and crate[1] == new_y:
                crate_idx = i
                break
        
        if crate_idx >= 0:
            # There is a crate, so check if it can be pushed
            new_crate_x, new_crate_y = new_x + dx, new_y + dy
            
            # Check if new crate position is valid
            if (new_crate_y < 0 or new_crate_y >= len(self.grid) or 
                new_crate_x < 0 or new_crate_x >= len(self.grid[0])):
                return False
            
            # Check if new crate position has a wall
            if self.grid[new_crate_y][new_crate_x] == WALL:
                return False
            
            # Check if new crate position has another crate
            for crate in self.crates_pos:
                if crate[0] == new_crate_x and crate[1] == new_crate_y:
                    return False
        
        return True
    
    def move_player(self, dx, dy):
        """Move the player and possibly push crates"""
        if not self.is_valid_move(dx, dy):
            return False
        
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
        
        # Save the move for undo functionality
        old_state = {
            'player_pos': self.player_pos[:],
            'crates_pos': [crate[:] for crate in self.crates_pos],
            'move_count': self.move_count
        }
        self.moves_history.append(old_state)
        
        # Check if there's a crate to push
        crate_idx = -1
        for i, crate in enumerate(self.crates_pos):
            if crate[0] == new_x and crate[1] == new_y:
                crate_idx = i
                break
        
        if crate_idx >= 0:
            # Push the crate
            self.crates_pos[crate_idx][0] += dx
            self.crates_pos[crate_idx][1] += dy
        
        # Move the player
        self.player_pos[0] = new_x
        self.player_pos[1] = new_y
        self.move_count += 1
        
        # Check if level is completed
        if self.is_level_completed():
            print(f"Level {self.level_num} completed in {self.move_count} moves!")
            self.level_num += 1
            self.load_level(self.level_num)
        
        return True
    
    def undo_move(self):
        """Undo the last move"""
        if not self.moves_history:
            return False
        
        last_state = self.moves_history.pop()
        self.player_pos = last_state['player_pos']
        self.crates_pos = last_state['crates_pos']
        self.move_count = last_state['move_count']
        
        return True
    
    def is_level_completed(self):
        """Check if all crates are on targets"""
        for crate in self.crates_pos:
            if crate not in self.targets_pos:
                return False
        return True
    
    def toggle_auto_solve(self):
        """Toggle automatic solving on/off"""
        if not self.auto_solve:
            # Start auto-solving
            print("Starting auto-solve with simulated annealing...")
            current_state = self.get_current_state()
            self.solver = SimulatedAnnealingSolver(current_state)
            self.solution_moves = self.solver.solve()
            
            if self.solution_moves:
                self.auto_solve = True
                self.current_move_index = 0
                print(f"Solution found with {len(self.solution_moves)} moves!")
            else:
                print("No solution found!")
        else:
            # Stop auto-solving
            self.auto_solve = False
            print("Auto-solving stopped")
    
    def auto_solve_step(self):
        """Perform one step of automatic solving"""
        if not self.auto_solve or not self.solution_moves:
            return False
        
        current_time = time.time()
        if current_time - self.last_auto_move_time < self.auto_solve_delay:
            return False
        
        self.last_auto_move_time = current_time
        
        if self.current_move_index < len(self.solution_moves):
            dx, dy = self.solution_moves[self.current_move_index]
            success = self.move_player(dx, dy)
            if success:
                self.current_move_index += 1
            return success
        else:
            self.auto_solve = False
            print("Auto-solve completed!")
            return False
    
    def draw(self):
        """Draw the game state to the screen"""
        # Clear the screen
        screen.fill(BLACK)
        
        # Draw the grid
        board_width = len(self.grid[0]) * TILE_SIZE
        board_height = len(self.grid) * TILE_SIZE
        board_offset_x = (SCREEN_WIDTH - board_width) // 2
        board_offset_y = (SCREEN_HEIGHT - board_height) // 2
        
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                pos_x = board_offset_x + x * TILE_SIZE
                pos_y = board_offset_y + y * TILE_SIZE
                
                # Draw floor
                pygame.draw.rect(screen, DARK_GRAY, (pos_x, pos_y, TILE_SIZE, TILE_SIZE))
                
                # Draw walls
                if cell == WALL:
                    pygame.draw.rect(screen, GRAY, (pos_x, pos_y, TILE_SIZE, TILE_SIZE))
        
        # Draw targets
        for target in self.targets_pos:
            target_x = board_offset_x + target[0] * TILE_SIZE
            target_y = board_offset_y + target[1] * TILE_SIZE
            pygame.draw.rect(screen, GREEN, (target_x, target_y, TILE_SIZE, TILE_SIZE))
        
        # Draw crates
        for crate in self.crates_pos:
            crate_x = board_offset_x + crate[0] * TILE_SIZE
            crate_y = board_offset_y + crate[1] * TILE_SIZE
            
            # Check if crate is on target
            if crate in self.targets_pos:
                pygame.draw.rect(screen, YELLOW, (crate_x, crate_y, TILE_SIZE, TILE_SIZE))
            else:
                pygame.draw.rect(screen, ORANGE, (crate_x, crate_y, TILE_SIZE, TILE_SIZE))
            
            # Draw crate inner detail
            pygame.draw.rect(screen, DARK_GRAY, (crate_x + 5, crate_y + 5, TILE_SIZE - 10, TILE_SIZE - 10))
        
        # Draw player
        player_x = board_offset_x + self.player_pos[0] * TILE_SIZE
        player_y = board_offset_y + self.player_pos[1] * TILE_SIZE
        pygame.draw.circle(screen, BLUE, (player_x + TILE_SIZE // 2, player_y + TILE_SIZE // 2), TILE_SIZE // 2 - 5)
        
        # Draw UI
        # Level info
        level_text = font.render(f"Level: {self.level_num}", True, WHITE)
        screen.blit(level_text, (20, 20))
        
        # Move count
        move_text = font.render(f"Moves: {self.move_count}", True, WHITE)
        screen.blit(move_text, (20, 50))
        
        # Auto-solve status
        auto_text = font.render(f"Auto-solve: {'ON' if self.auto_solve else 'OFF'}", True, WHITE)
        screen.blit(auto_text, (SCREEN_WIDTH - 180, 20))
        
        if self.auto_solve and self.solution_moves:
            progress_text = font.render(f"Progress: {self.current_move_index}/{len(self.solution_moves)}", True, WHITE)
            screen.blit(progress_text, (SCREEN_WIDTH - 180, 50))
        
        # Controls info
        controls_text = small_font.render("Arrow Keys: Move, S: Auto-solve, R: Restart, Z: Undo", True, WHITE)
        screen.blit(controls_text, (20, SCREEN_HEIGHT - 30))
        
        # Update the display
        pygame.display.flip()

def main():
    game = BoxBangGame()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    game.move_player(0, -1)
                elif event.key == pygame.K_RIGHT:
                    game.move_player(1, 0)
                elif event.key == pygame.K_DOWN:
                    game.move_player(0, 1)
                elif event.key == pygame.K_LEFT:
                    game.move_player(-1, 0)
                elif event.key == pygame.K_r:  # Restart level
                    game.load_level(game.level_num)
                elif event.key == pygame.K_z:  # Undo
                    game.undo_move()
                elif event.key == pygame.K_s:  # Toggle auto-solve
                    game.toggle_auto_solve()
        
        # Auto-solve step if enabled
        if game.auto_solve:
            game.auto_solve_step()
        
        # Draw the game
        game.draw()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()