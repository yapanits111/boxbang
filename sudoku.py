import pygame
import random
import math
import time

# Update the constants at the top of the file
WIDTH, HEIGHT = 600, 700
CELL_SIZE = 60
GRID_SIZE = CELL_SIZE * 9

pygame.init()

# Create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sudoku")

# Update fonts and colors
FONT = pygame.font.SysFont("arial", 40)
SMALL_FONT = pygame.font.SysFont("arial", 24)
TITLE_FONT = pygame.font.SysFont("arial", 48, bold=True)

# Updated colors for better contrast and modern look
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (128, 128, 128)
LIGHT_BLUE = (176, 224, 230)
DARK_BLUE = (0, 0, 139)
RED = (220, 20, 60)
GREEN = (34, 139, 34)
BG_COLOR = (240, 248, 255)
HINT_BTN_COLOR = (70, 130, 180)
NEW_GAME_BTN_COLOR = (65, 105, 225)
GOLD = (255, 215, 0)
VICTORY_BG = (0, 100, 0)

def is_valid(board, row, col, num):
    # Check row
    for i in range(9):
        if board[row][i] == num:
            return False
    
    # Check column
    for i in range(9):
        if board[i][col] == num:
            return False
    
    # Check 3x3 box
    box_x, box_y = (row // 3) * 3, (col // 3) * 3
    for i in range(box_x, box_x + 3):
        for j in range(box_y, box_y + 3):
            if board[i][j] == num:
                return False
    
    return True

def is_valid_move(board, row, col, num):
    # Temporarily store the original value
    original = board[row][col]
    board[row][col] = 0  # Clear the cell to check validity
    
    valid = is_valid(board, row, col, num)
    
    # Restore the original value
    board[row][col] = original
    
    return valid

def solve_board(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_board(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def generate_puzzle():
    # Start with an empty board
    board = [[0]*9 for _ in range(9)]
    
    # Fill diagonal 3x3 boxes first (they don't interfere with each other)
    for box in range(3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box*3 + i][box*3 + j] = nums[i*3 + j]
    
    # Solve the rest of the board
    solve_board(board)
    
    # Store the complete solution
    solution = [row[:] for row in board]
    
    # Remove numbers to create puzzle (adjust difficulty by changing the number removed)
    cells_to_remove = 40  # Adjust this for difficulty
    cells = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(cells)
    
    for i, j in cells[:cells_to_remove]:
        board[i][j] = 0
    
    return board, solution

def get_hint(selected, puzzle, solution):
    if not selected:
        return None, "Select a cell first."
    
    i, j = selected
    
    if puzzle[i][j] != 0:
        return None, "Cell is already filled."
    
    # Return the correct number from the solution
    return (i, j, solution[i][j]), "Hint provided!"

def check_solution(puzzle, solution):
    for i in range(9):
        for j in range(9):
            if puzzle[i][j] != solution[i][j]:
                return False
    return True

def draw_board(selected, puzzle, original_puzzle, status_message):
    screen.fill(BG_COLOR)
    
    # Draw title
    title = TITLE_FONT.render("SUDOKU", True, DARK_BLUE)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))

    # Center the grid
    grid_start_x = (WIDTH - GRID_SIZE) // 2
    grid_start_y = 80

    # Draw the grid background
    pygame.draw.rect(screen, WHITE, (grid_start_x, grid_start_y, GRID_SIZE, GRID_SIZE))

    # Draw the grid lines
    for i in range(10):
        thick = 4 if i % 3 == 0 else 1
        # Vertical lines
        pygame.draw.line(screen, BLACK, 
                        (grid_start_x + i * CELL_SIZE, grid_start_y),
                        (grid_start_x + i * CELL_SIZE, grid_start_y + GRID_SIZE), thick)
        # Horizontal lines
        pygame.draw.line(screen, BLACK,
                        (grid_start_x, grid_start_y + i * CELL_SIZE),
                        (grid_start_x + GRID_SIZE, grid_start_y + i * CELL_SIZE), thick)

    # Draw numbers
    for i in range(9):
        for j in range(9):
            num = puzzle[i][j]
            if num != 0:
                color = BLACK if original_puzzle[i][j] != 0 else DARK_BLUE
                text = FONT.render(str(num), True, color)
                x = grid_start_x + j * CELL_SIZE + (CELL_SIZE - text.get_width()) // 2
                y = grid_start_y + i * CELL_SIZE + (CELL_SIZE - text.get_height()) // 2
                screen.blit(text, (x, y))

    # Highlight selected cell
    if selected:
        pygame.draw.rect(screen, RED,
                        (grid_start_x + selected[1] * CELL_SIZE,
                         grid_start_y + selected[0] * CELL_SIZE,
                         CELL_SIZE, CELL_SIZE), 3)

    # Draw buttons
    button_y = grid_start_y + GRID_SIZE + 30
    
    # New Game button
    new_game_btn = pygame.Rect(WIDTH//4 - 70, button_y, 140, 50)
    pygame.draw.rect(screen, NEW_GAME_BTN_COLOR, new_game_btn, border_radius=10)
    new_game_text = SMALL_FONT.render("New Game", True, WHITE)
    screen.blit(new_game_text, (new_game_btn.centerx - new_game_text.get_width()//2,
                               new_game_btn.centery - new_game_text.get_height()//2))

    # Hint button
    hint_btn = pygame.Rect(3*WIDTH//4 - 70, button_y, 140, 50)
    pygame.draw.rect(screen, HINT_BTN_COLOR, hint_btn, border_radius=10)
    hint_text = SMALL_FONT.render("Get Hint", True, WHITE)
    screen.blit(hint_text, (hint_btn.centerx - hint_text.get_width()//2,
                           hint_btn.centery - hint_text.get_height()//2))

    # Draw status message with special celebration for victory
    if status_message:
        if "Congratulations" in status_message:
            # Draw celebration background
            celebration_bg = pygame.Rect(0, button_y + 60, WIDTH, 80)
            pygame.draw.rect(screen, VICTORY_BG, celebration_bg)
            
            # Draw multiple celebration messages
            main_text = SMALL_FONT.render("🎉 Congratulations! You solved the puzzle! 🎉", True, GOLD)
            sub_text = SMALL_FONT.render("You're a Sudoku Master!", True, WHITE)
            
            screen.blit(main_text, (WIDTH//2 - main_text.get_width()//2, button_y + 70))
            screen.blit(sub_text, (WIDTH//2 - sub_text.get_width()//2, button_y + 100))
        else:
            color = DARK_BLUE if "Hint" in status_message else RED
            status_text = SMALL_FONT.render(status_message, True, color)
            screen.blit(status_text, (WIDTH//2 - status_text.get_width()//2, button_y + 70))

    pygame.display.update()

def main():
    # Initialize game state
    puzzle, solution = generate_puzzle()
    original_puzzle = [row[:] for row in puzzle]
    status_message = "Welcome to Sudoku! Select a cell and enter a number. Press H for hint, N for new game."
    
    selected = None
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                grid_start_x = (WIDTH - GRID_SIZE) // 2
                grid_start_y = 80
                button_y = grid_start_y + GRID_SIZE + 30

                # Click within puzzle grid
                if (grid_start_x <= x <= grid_start_x + GRID_SIZE and 
                    grid_start_y <= y <= grid_start_y + GRID_SIZE):
                    row = (y - grid_start_y) // CELL_SIZE
                    col = (x - grid_start_x) // CELL_SIZE
                    if 0 <= row < 9 and 0 <= col < 9:
                        selected = (row, col)
                        status_message = f"Selected cell ({row+1}, {col+1})"

                # Hint button
                hint_btn = pygame.Rect(3*WIDTH//4 - 70, button_y, 140, 50)
                if hint_btn.collidepoint(x, y):
                    hint, msg = get_hint(selected, puzzle, solution)
                    if hint:
                        i, j, val = hint
                        puzzle[i][j] = val
                        status_message = f"Hint: {val} placed at ({i+1},{j+1})"
                        if check_solution(puzzle, solution):
                            status_message = "Congratulations! You solved the puzzle!"
                    else:
                        status_message = msg

                # New game button
                new_game_btn = pygame.Rect(WIDTH//4 - 70, button_y, 140, 50)
                if new_game_btn.collidepoint(x, y):
                    puzzle, solution = generate_puzzle()
                    original_puzzle = [row[:] for row in puzzle]
                    selected = None
                    status_message = "New game started!"

            elif event.type == pygame.KEYDOWN:
                # Number input (1-9)
                if selected and event.key in range(pygame.K_1, pygame.K_9+1):
                    if original_puzzle[selected[0]][selected[1]] == 0:
                        val = event.key - pygame.K_0
                        
                        # Check if the move is valid before placing it
                        if is_valid_move(puzzle, selected[0], selected[1], val):
                            puzzle[selected[0]][selected[1]] = val
                            status_message = f"Good move! {val} placed at ({selected[0]+1},{selected[1]+1})"
                            
                            # Check if puzzle is solved
                            if check_solution(puzzle, solution):
                                status_message = "Congratulations! You solved the puzzle!"
                        else:
                            # Place the number anyway but show warning
                            puzzle[selected[0]][selected[1]] = val
                            status_message = f"Invalid move! {val} conflicts with existing numbers"
                    else:
                        status_message = "Cannot modify original puzzle numbers!"
                
                # Clear cell (Backspace/Delete)
                elif event.key == pygame.K_BACKSPACE or event.key == pygame.K_DELETE:
                    if selected and original_puzzle[selected[0]][selected[1]] == 0:
                        puzzle[selected[0]][selected[1]] = 0
                        status_message = f"Cleared cell ({selected[0]+1},{selected[1]+1})"
                
                # Hint shortcut (H key)
                elif event.key == pygame.K_h:
                    hint, msg = get_hint(selected, puzzle, solution)
                    if hint:
                        i, j, val = hint
                        puzzle[i][j] = val
                        status_message = f"Hint: {val} placed at ({i+1},{j+1})"
                        if check_solution(puzzle, solution):
                            status_message = "Congratulations! You solved the puzzle!"
                    else:
                        status_message = msg
                
                # New game shortcut (N key)
                elif event.key == pygame.K_n:
                    puzzle, solution = generate_puzzle()
                    original_puzzle = [row[:] for row in puzzle]
                    selected = None
                    status_message = "New game started! (Press N for new game, H for hint)"

        draw_board(selected, puzzle, original_puzzle, status_message)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
