import pygame
import sys
import os
from collections import deque

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
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)

# Game elements
WALL = '#'
PLAYER = '@'
CRATE = '$'
TARGET = '.'
FLOOR = ' '

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()

# Font
font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 16)
large_font = pygame.font.SysFont('Arial', 36)

class GameState:
    def __init__(self, level_grid, player_pos, crates_pos, targets_pos, move_count=0):
        self.grid = level_grid
        self.player_pos = player_pos[:]
        self.crates_pos = [crate[:] for crate in crates_pos]
        self.targets_pos = targets_pos[:]
        self.move_count = move_count
    
    def copy(self):
        return GameState(
            [row[:] for row in self.grid],
            self.player_pos[:],
            [crate[:] for crate in self.crates_pos],
            self.targets_pos[:],
            self.move_count
        )
    
    def is_solved(self):
        """Check if all crates are on targets"""
        return len(self.crates_pos) == len(self.targets_pos) and all(crate in self.targets_pos for crate in self.crates_pos)
    
    def get_hash(self):
        """Get a hash of the current state for duplicate detection"""
        crates_tuple = tuple(tuple(crate) for crate in sorted(self.crates_pos))
        return (tuple(self.player_pos), crates_tuple)
    
    def get_possible_moves(self):
        """Get all possible moves from current state"""
        moves = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        
        for dx, dy in directions:
            if self._is_valid_move(dx, dy):
                moves.append((dx, dy))
        
        return moves
    
    def _is_valid_move(self, dx, dy):
        """Check if a move is valid"""
        new_x, new_y = self.player_pos[0] + dx, self.player_pos[1] + dy
        
        # Check bounds
        if not (0 <= new_x < len(self.grid[0]) and 0 <= new_y < len(self.grid)):
            return False
        
        # Check wall
        if self.grid[new_y][new_x] == WALL:
            return False
        
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
            if not (0 <= new_crate_x < len(self.grid[0]) and 0 <= new_crate_y < len(self.grid)):
                return False
            
            # Check wall for crate
            if self.grid[new_crate_y][new_crate_x] == WALL:
                return False
            
            # Check if another crate is there
            for other_crate in self.crates_pos:
                if other_crate[0] == new_crate_x and other_crate[1] == new_crate_y:
                    return False
        
        return True
    
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

class BFSSolver:
    def __init__(self, initial_state, max_moves=100):
        self.initial_state = initial_state
        self.max_moves = max_moves
    
    def solve(self):
        """Solve using BFS to find shortest solution"""
        queue = deque([(self.initial_state, [])])
        visited = {self.initial_state.get_hash()}
        
        print("Starting BFS solver...")
        
        while queue:
            current_state, path = queue.popleft()
            
            # Check if solved
            if current_state.is_solved():
                print(f"Solution found with {len(path)} moves!")
                return path
            
            # Don't explore paths that are too long
            if len(path) >= self.max_moves:
                continue
            
            # Try all possible moves
            for move in current_state.get_possible_moves():
                new_state = current_state.apply_move(move[0], move[1])
                state_hash = new_state.get_hash()
                
                if state_hash not in visited:
                    visited.add(state_hash)
                    new_path = path + [move]
                    queue.append((new_state, new_path))
        
        print("No solution found within move limit")
        return None

class BoxBangGame:
    def __init__(self):
        self.level_num = 1
        self.auto_solve = False
        self.solver = None
        self.solution_moves = []
        self.current_move_index = 0
        self.show_level_select = False
        self.available_levels = self.scan_available_levels()
        self.load_level(self.level_num)
    
    def scan_available_levels(self):
        """Scan for available level files"""
        levels = []
        for i in range(1, 101):  # Check for levels 1-100
            if os.path.exists(f"lvl{i}.txt"):
                levels.append(i)
        
        # If no level files found, create a default level
        if not levels:
            self.create_default_level()
            levels = [1]
        
        return levels
    
    def create_default_level(self):
        """Create a simple default level"""
        default_level = [
            "########",
            "#  .   #",
            "#  $@  #",
            "#      #",
            "########"
        ]
        
        with open("lvl1.txt", "w") as f:
            f.write('\n'.join(default_level))
    
    def load_level(self, level_num):
        """Load a level from file"""
        filename = f"lvl{level_num}.txt"
        
        if not os.path.exists(filename):
            print(f"Level {level_num} not found!")
            return False
        
        try:
            with open(filename, "r") as f:
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
                    else:
                        grid_row.append(cell)
                self.grid.append(grid_row)
            
            self.level_num = level_num
            self.move_count = 0
            self.moves_history = []
            
            # Reset auto-solve state
            self.auto_solve = False
            self.solver = None
            self.solution_moves = []
            self.current_move_index = 0
            
            print(f"Loaded level {level_num}")
            return True
            
        except Exception as e:
            print(f"Error loading level {level_num}: {e}")
            return False
    
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
        return self.get_current_state()._is_valid_move(dx, dy)
    
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
        return all(crate in self.targets_pos for crate in self.crates_pos)
    
    def toggle_auto_solve(self):
        """Toggle automatic solving on/off"""
        if not self.auto_solve:
            # Start auto-solving
            print("Starting auto-solve with BFS...")
            current_state = self.get_current_state()
            self.solver = BFSSolver(current_state)
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
    
    def next_level(self):
        """Load next level if available"""
        current_index = self.available_levels.index(self.level_num) if self.level_num in self.available_levels else 0
        if current_index < len(self.available_levels) - 1:
            self.load_level(self.available_levels[current_index + 1])
            return True
        return False

    def previous_level(self):
        """Load previous level if available"""
        current_index = self.available_levels.index(self.level_num) if self.level_num in self.available_levels else 0
        if current_index > 0:
            self.load_level(self.available_levels[current_index - 1])
            return True
        return False
    
    def toggle_level_select(self):
        """Toggle level selection screen"""
        self.show_level_select = not self.show_level_select
    
    def select_level_from_number(self, level_num):
        """Select a specific level by number"""
        if level_num in self.available_levels:
            self.load_level(level_num)
            self.show_level_select = False
        else:
            print(f"Level {level_num} not available!")
    
    def draw_level_select(self):
        """Draw the level selection screen"""
        screen.fill(BLACK)
        
        # Title
        title_text = large_font.render("Level Selection", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title_text, title_rect)
        
        # Current level indicator
        current_text = font.render(f"Current Level: {self.level_num}", True, YELLOW)
        current_rect = current_text.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(current_text, current_rect)
        
        # Available levels
        levels_text = font.render("Available Levels:", True, WHITE)
        screen.blit(levels_text, (50, 150))
        
        # Draw level grid
        cols = 10
        start_x = 50
        start_y = 200
        cell_width = 70
        cell_height = 40
        
        for i, level in enumerate(self.available_levels):
            row = i // cols
            col = i % cols
            x = start_x + col * cell_width
            y = start_y + row * cell_height
            
            # Highlight current level
            if level == self.level_num:
                pygame.draw.rect(screen, YELLOW, (x, y, cell_width - 5, cell_height - 5))
                text_color = BLACK
            else:
                pygame.draw.rect(screen, GRAY, (x, y, cell_width - 5, cell_height - 5))
                text_color = WHITE
            
            # Draw level number
            level_text = font.render(str(level), True, text_color)
            text_rect = level_text.get_rect(center=(x + cell_width // 2, y + cell_height // 2))
            screen.blit(level_text, text_rect)
        
        # Instructions
        instructions = [
            "Use number keys (1-9, 0) to select levels",
            "Press Escape to return to game",
            "Use Page Up/Down to navigate levels"
        ]
        
        for i, instruction in enumerate(instructions):
            text = small_font.render(instruction, True, WHITE)
            screen.blit(text, (50, 450 + i * 20))
    
    def draw(self):
        """Draw the game state to the screen"""
        if self.show_level_select:
            self.draw_level_select()
            return
        
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
        
        # Level completion status
        if self.is_level_completed():
            completed_text = font.render("LEVEL COMPLETED!", True, GREEN)
            screen.blit(completed_text, (20, 80))
        
        # Auto-solve status
        if self.auto_solve:
            auto_text = font.render(f"Auto-solving: {self.current_move_index}/{len(self.solution_moves)}", True, GREEN)
            screen.blit(auto_text, (SCREEN_WIDTH - 200, 20))
        
        # Controls info
        controls_text = small_font.render("Arrow Keys: Move | S: Auto-solve | R: Restart | Z: Undo | L: Level Select", True, WHITE)
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
                if event.key == pygame.K_ESCAPE:
                    if game.show_level_select:
                        game.show_level_select = False
                    else:
                        running = False
                elif event.key == pygame.K_LEFT:
                    game.move_player(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    game.move_player(1, 0)
                elif event.key == pygame.K_UP:
                    game.move_player(0, -1)
                elif event.key == pygame.K_DOWN:
                    game.move_player(0, 1)
                elif event.key == pygame.K_r:
                    game.load_level(game.level_num)
                elif event.key == pygame.K_s:
                    game.toggle_auto_solve()
                elif event.key == pygame.K_z:
                    game.undo_move()
                elif event.key == pygame.K_l:
                    game.toggle_level_select()
                elif event.key == pygame.K_PAGEUP:
                    game.previous_level()
                elif event.key == pygame.K_PAGEDOWN:
                    game.next_level()
                elif event.key == pygame.K_n:
                    game.next_level()
                elif event.key == pygame.K_p:
                    game.previous_level()
                # Number keys for quick level select
                elif event.key in range(pygame.K_0, pygame.K_9 + 1):
                    level = event.key - pygame.K_0 if event.key != pygame.K_0 else 10
                    game.select_level_from_number(level)

        # Update game state
        if game.auto_solve:
            game.auto_solve_step()

        # Draw everything
        game.draw()
        
        # Control game speed
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()