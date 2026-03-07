#   __.-._
#   '-._"7'
#    /'.-c
#    |  //
#   _)_/||
#
# chess_view.py - A comprehensive chess PGN viewer and replayer with detailed logging.
# Author: santakd
# Contact: santakd at gmail dot com
# Date: March 05, 2026
# Version: 1.0.8
# License: MIT License 

# pip install pygame    # For rendering the chessboard and GUI
# pip install chess     # For parsing PGN files and managing chess game state
# pip install tkinter   # Usually pre-installed with Python, for file dialog

import os                           # For file handling and environment variables
import pygame                       # For rendering the chessboard and GUI
import chess                        # For parsing PGN files and managing chess game state
import chess.pgn                    # For parsing PGN files and extracting moves and comments
import logging                      # For logging events, errors, and debugging information
import sys                          # For exiting the application on critical errors
import time                         # For performance measurement and debugging
import io                           # For parsing PGN from string
from tkinter import filedialog, Tk  # For file browser dialog
from tkinter import messagebox      # For user-friendly error messages


# Constants for GUI layout and behavior
SCREEN_WIDTH = 1080                             # Changed to 1080 as requested
BOARD_SIZE = 640                                # Size of the chessboard (640x640 for better visibility)    
SQUARE_SIZE = BOARD_SIZE // 8                   # Size of each square on the chessboard
SIDE_PANEL_WIDTH = SCREEN_WIDTH - BOARD_SIZE    # Width of the side panel for PGN display and controls
FPS = 60                                        # Frames per second for the Pygame loop, can be adjusted for performance
LINE_HEIGHT = 20                                # Height of each line of PGN text in the side panel, can be adjusted based on font size
PGN_AREA_HEIGHT = SCREEN_WIDTH - 150            # Height of PGN display area (adjust as needed)

# Colors
LIGHT_SQUARE = (233, 214, 176)      # Light board square color
DARK_SQUARE = (164, 128, 92)        # Dark board square color
TEXT_COLOR = (0, 0, 0)              # Text color
BUTTON_COLOR = (150, 150, 150)      # Button background
BUTTON_HOVER = (200, 200, 200)      # Button hover
HIGHLIGHT_COLOR = (255, 0, 0)       # Red for highlighting current move
HIGHLIGHT_BG = (255, 255, 200)      # Light yellow background for better visibility

ENABLE_HIGHLIGHTING = True          # Set to False to disable highlighting

GENERATE_LOG = True  # Generate timestamped log file

# Class for the PGN Replayer application
class PGNReplayer:
    def __init__(self):
        """Initialize Pygame, load resources, and set up game state."""
        try:
            # Initialize Tkinter first to avoid conflicts with Pygame on macOS
            os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Suppress Tk deprecation warnings on macOS
            self.tk_root = Tk()
            self.tk_root.withdraw()
            self.tk_root.update()  # Force Tk update to initialize properly before Pygame

            # Workaround for macOS SDL/Cocoa issues: Set environment variables before pygame.init()
            os.environ['SDL_VIDEO_DRIVER'] = 'cocoa'  # Force Cocoa driver on macOS
            os.environ['SDL_VIDEO_CENTERED'] = '1'    # Center the window
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (100, 100)  # Set fixed position to avoid display issues
            os.environ['PYGAME_BLEND_ALPHA_SDL2'] = '1'  # Enable alpha blending for SDL2

            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
            pygame.display.set_caption("Chess PGN Replayer")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)  # Font for buttons and labels
            self.small_font = pygame.font.SysFont(None, 18)  # Smaller font for PGN text to fit more lines

            self.setup_logging()

            # Load piece images from 'pieces' directory
            self.load_piece_images()

            # Game state variables
            self.game = None  # Parsed chess.pgn.Game object
            self.moves = []   # List of chess.Move objects
            self.sans = []    # List of SAN strings (move notations)
            self.board = chess.Board()  # Current board state for replay
            self.current_index = -1  # Current move index (-1 = starting position before first move)
            self.pgn_text = ""  # Full PGN string for display
            self.pgn_lines = []  # Split lines for rendering (wrapped)

            # Button rectangles
            self.load_button = pygame.Rect(BOARD_SIZE + 10, 10, SIDE_PANEL_WIDTH - 20, 30)  # Load PGN
            self.rewind_start_button = pygame.Rect(BOARD_SIZE + 10, 50, (SIDE_PANEL_WIDTH - 20) / 4, 30)  # <<
            self.rewind_button = pygame.Rect(BOARD_SIZE + 10 + (SIDE_PANEL_WIDTH - 20) / 4, 50, (SIDE_PANEL_WIDTH - 20) / 4, 30)  # <
            self.forward_button = pygame.Rect(BOARD_SIZE + 10 + 2 * (SIDE_PANEL_WIDTH - 20) / 4, 50, (SIDE_PANEL_WIDTH - 20) / 4, 30)  # >
            self.forward_end_button = pygame.Rect(BOARD_SIZE + 10 + 3 * (SIDE_PANEL_WIDTH - 20) / 4, 50, (SIDE_PANEL_WIDTH - 20) / 4, 30)  # >>
            self.quit_button = pygame.Rect(BOARD_SIZE + 10, SCREEN_WIDTH - 40, SIDE_PANEL_WIDTH - 20, 30)  # Quit

            # PGN display area (scrollable if needed, but for simplicity, we render visible lines)
            self.pgn_area_y = 100  # Starting y-position for PGN text
            self.pgn_scroll_offset = 0  # For scrolling long PGN

            logging.info("Chess View for PGN Replay initialized successfully.")

        except pygame.error as e:
            logging.error(f"Pygame initialization failed: {e}")
            messagebox.showerror("Initialization Error", "Failed to initialize Pygame. Check your installation.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Unexpected error during initialization: {e}")
            messagebox.showerror("Initialization Error", "An unexpected error occurred. See log for details.")
            sys.exit(1)

    def setup_logging(self):
        """Setup timestamped logging for the game"""
        self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        if GENERATE_LOG:
            log_file = f"chess_view_{self.timestamp}.log"
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout)
                ],
                force=True  # Force reconfiguration even if handlers exist
            )
            logging.info(f"Logging to: {log_file}")
        else:
            logging.basicConfig(level=logging.WARNING)  # Minimal logging if disabled


    def load_piece_images(self):
        """Load chess piece images from 'pieces' directory."""
        try:
            self.pieces_images = {}
            for color in ['white', 'black']:
                for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
                    symbol = (piece[0].upper() if piece != 'knight' else 'N') if color == 'white' else (piece[0].lower() if piece != 'knight' else 'n')
                    img_path = f'pieces/{color}_{piece}.png'
                    if os.path.exists(img_path):
                        img = pygame.image.load(img_path)
                        self.pieces_images[chess.Piece.from_symbol(symbol)] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
                        logging.debug(f"Loaded image: {img_path}")
                    else:
                        logging.warning(f"Missing image: {img_path}. Pieces may not display correctly.")
        except Exception as e:
            logging.error(f"Error loading piece images: {e}")
            messagebox.showwarning("Image Load Warning", "Some piece images could not be loaded. Check 'pieces' directory.")


    def run(self):
        """Main loop for handling events, updates, and rendering."""
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.load_button.collidepoint(event.pos):
                        self.load_pgn_file()
                    elif self.rewind_start_button.collidepoint(event.pos):
                        self.rewind_to_start()
                    elif self.rewind_button.collidepoint(event.pos):
                        self.rewind_move()
                    elif self.forward_button.collidepoint(event.pos):
                        self.forward_move()
                    elif self.forward_end_button.collidepoint(event.pos):
                        self.forward_to_end()
                    elif self.quit_button.collidepoint(event.pos):
                        running = False
                if event.type == pygame.MOUSEWHEEL:
                    # Scroll PGN text
                    self.pgn_scroll_offset += event.y * LINE_HEIGHT  # Scroll up/down
                    self.pgn_scroll_offset = min(0, self.pgn_scroll_offset)

            self.screen.fill((255, 255, 255))  # Clear screen
            self.draw_board()
            self.draw_pieces()
            self.draw_side_panel()
            self.draw_buttons(mouse_pos)
            self.draw_pgn_text()

            pygame.display.flip()
            self.clock.tick(FPS)

            if self.tk_root:
                try:
                    self.tk_root.update()
                except:
                    pass

        logging.info("Application exited.")
        logging.shutdown()
        pygame.quit()


    def load_pgn_file(self):
        """Browse and load a PGN file, parse it, and reset game state."""
        try:
            file_path = filedialog.askopenfilename(
                title="Select PGN File",
                filetypes=[("PGN Files", "*.pgn"), ("All Files", "*.*")]
            )
            if not file_path:
                logging.info("PGN file selection canceled.")
                return

            logging.info(f"Loading PGN from: {file_path}")
            with open(file_path, 'r') as f:
                pgn_content = f.read()

            # Parse PGN (handles {eval: xx} as comments, which are ignored in move extraction)
            pgn = io.StringIO(pgn_content)
            self.game = chess.pgn.read_game(pgn)
            if self.game is None:
                raise ValueError("Invalid PGN format.")

            # Extract moves and SANs
            self.moves = []
            self.sans = []
            node = self.game
            board_temp = chess.Board()  # Temp board for SAN generation
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                san = board_temp.san(move)
                self.moves.append(move)
                self.sans.append(san)
                board_temp.push(move)
                node = next_node

            # Reset to starting position
            self.board = chess.Board()
            self.current_index = -1  # -1 indicates before first move

            # Store full PGN text for display (including headers and comments)
            self.pgn_text = pgn_content
            self.wrap_pgn_lines()

            logging.info(f"PGN loaded: {len(self.moves)} moves.")

        except FileNotFoundError:
            logging.error("Selected PGN file not found.")
            messagebox.showerror("File Error", "PGN file not found.")
        except ValueError as ve:
            logging.error(f"PGN parsing error: {ve}")
            messagebox.showerror("PGN Error", "Invalid PGN format.")
        except Exception as e:
            logging.error(f"Unexpected error loading PGN: {e}")
            messagebox.showerror("Load Error", "An unexpected error occurred while loading the PGN.")


    def wrap_pgn_lines(self):
        """Wrap long PGN lines to fit within the side panel width and track move positions."""
        self.pgn_lines = []
        self.wrapped_move_positions = []  # List of (line_idx, start_x, end_x) for each move

        move_counter = 0  # Tracks the current move index for sans
        lines = self.pgn_text.splitlines()
        for line in lines:
            # Skip wrapping for headers, but add them
            if line.strip().startswith('['):
                self.pgn_lines.append(line)
                continue

            # Process move lines
            words = line.split()
            current_line = ""
            current_x = 0
            line_idx = len(self.pgn_lines)

            for word in words:
                # Ignore comments like {eval: x.xx}
                if word.startswith('{') and word.endswith('}'):
                    continue  # Skip comments entirely

                test = current_line + (" " if current_line else "") + word
                test_width = self.small_font.size(test)[0]

                if test_width > SIDE_PANEL_WIDTH - 30:
                    # Append the current line and start a new one
                    self.pgn_lines.append(current_line)
                    line_idx += 1
                    current_line = word
                    current_x = 0
                else:
                    if current_line:
                        current_x += self.small_font.size(" ")[0]
                    start_x = current_x
                    current_x += self.small_font.size(word)[0]
                    current_line = test

                    # Check if this word is a move (e.g., 1., e4, Nf6, but ignore numbers like 1.)
                    if not word.endswith('.') and move_counter < len(self.sans) and self.sans[move_counter] in word:
                        self.wrapped_move_positions.append((line_idx, start_x, current_x))
                        move_counter += 1

            if current_line:
                self.pgn_lines.append(current_line)

        logging.debug(f"Wrapped PGN into {len(self.pgn_lines)} lines. Tracked {len(self.wrapped_move_positions)} moves.")


    def forward_move(self):
        """Advance to the next move if available."""
        if self.current_index < len(self.moves) - 1:
            self.current_index += 1
            self.board.push(self.moves[self.current_index])
            logging.debug(f"Advanced to move {self.current_index + 1}: {self.sans[self.current_index]}")


    def rewind_move(self):
        """Go back to the previous move if possible."""
        if self.current_index >= 0:
            self.board.pop()
            self.current_index -= 1
            logging.debug(f"Rewound to move {self.current_index + 1}")


    def forward_to_end(self):
        """Fast-forward to the end of the game."""
        while self.current_index < len(self.moves) - 1:
            self.forward_move()
        logging.debug("Fast-forwarded to end of game.")


    def rewind_to_start(self):
        """Rewind to the starting position."""
        while self.current_index >= 0:
            self.rewind_move()
        logging.debug("Rewound to start position.")


    def draw_board(self):
        """Render the chessboard squares."""
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


    def draw_pieces(self):
        """Render pieces on the current board state."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                img = self.pieces_images.get(piece)
                if img:
                    col = chess.square_file(square)
                    row = 7 - chess.square_rank(square)
                    self.screen.blit(img, (col * SQUARE_SIZE, row * SQUARE_SIZE))


    def draw_side_panel(self):
        """Draw the side panel background."""
        pygame.draw.rect(self.screen, (200, 200, 200), (BOARD_SIZE, 0, SIDE_PANEL_WIDTH, SCREEN_WIDTH))


    def draw_buttons(self, mouse_pos):
        """Render buttons with hover effects."""
        buttons = [
            (self.load_button, "Load PGN"),
            (self.rewind_start_button, "<<"),
            (self.rewind_button, "<"),
            (self.forward_button, ">"),
            (self.forward_end_button, ">>"),
            (self.quit_button, "Quit")
        ]
        for rect, label in buttons:
            color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
            pygame.draw.rect(self.screen, color, rect)
            text = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(text, text.get_rect(center=rect.center))


    def draw_pgn_text(self):
        """Render the wrapped PGN text on the side panel with scrolling and highlighting."""
        y = self.pgn_area_y + self.pgn_scroll_offset
        if not self.pgn_lines:
            text = self.font.render("No PGN loaded", True, TEXT_COLOR)
            self.screen.blit(text, (BOARD_SIZE + 10, self.pgn_area_y))
            return

        highlight_move_idx = self.current_index if self.current_index >= 0 else -1

        for i, line in enumerate(self.pgn_lines):
            if y + LINE_HEIGHT < self.pgn_area_y or y > self.pgn_area_y + PGN_AREA_HEIGHT:
                y += LINE_HEIGHT
                continue

            text_surf = self.small_font.render(line, True, TEXT_COLOR)
            self.screen.blit(text_surf, (BOARD_SIZE + 10, y))

            if ENABLE_HIGHLIGHTING and highlight_move_idx >= 0 and highlight_move_idx < len(self.wrapped_move_positions):
                line_idx, start_x, end_x = self.wrapped_move_positions[highlight_move_idx]
                if line_idx == i:
                    pygame.draw.rect(
                        self.screen,
                        HIGHLIGHT_BG,
                        (BOARD_SIZE + 10 + start_x, y, end_x - start_x, LINE_HEIGHT)
                    )
                    move_word = self.sans[highlight_move_idx]
                    word_surf = self.small_font.render(move_word, True, HIGHLIGHT_COLOR)
                    self.screen.blit(word_surf, (BOARD_SIZE + 10 + start_x, y))

            y += LINE_HEIGHT

        total_height = len(self.pgn_lines) * LINE_HEIGHT
        max_offset = min(0, PGN_AREA_HEIGHT - total_height)
        self.pgn_scroll_offset = max(max_offset, self.pgn_scroll_offset)


# Main execution
if __name__ == "__main__":
    replayer = PGNReplayer()
    replayer.run()