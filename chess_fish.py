#   __.-._
#   '-._"7'
#    /'.-c
#    |  //
#   _)_/||
#
# chess_fish.py
# Author: santakd
# Contact: santakd at gmail dot com
# Date: February 15, 2026
# Version: 1.0.8  # Updated version with PGN validation and quit handling
# License: MIT License

# Required installations:
# pip install pygame
# pip install chess
# pip install stockfish             # This is for Stockfish engine integration, install stockfish engine via Homebrew and ensure path is correct

import pygame                       # For game UI and graphics
import chess                        # For chess logic and board management
import logging                      # For logging game events and Stockfish info
import time                         # For timing AI moves and timestamps
import sys                          # For system exits
import os                           # For file operations like creating directories
import io                           # For in-memory PGN parsing during validation
from math import inf                # For minimax infinity values
import datetime                     # Unused but kept for potential future use
import argparse                     # For command-line arguments parsing
from stockfish import Stockfish     # For integrating Stockfish AI engine

# Constants for game window and board
SCREEN_WIDTH = 800  # Total screen width
BOARD_SIZE = 640  # Chess board size (8x8 squares)
SQUARE_SIZE = BOARD_SIZE // 8  # Size of each square
SIDE_PANEL_WIDTH = SCREEN_WIDTH - BOARD_SIZE  # Width for side panel
FPS = 60  # Frames per second for smooth rendering

# Logging and output flags
GENERATE_LOG = True  # Generate timestamped log file
GENERATE_PGN = True  # Generate PGN file for games

# Stockfish configuration
USE_STOCKFISH = True  # Enable Stockfish for AI (fallback to minimax if False)
STOCKFISH_PATH = "/opt/homebrew/Cellar/stockfish/18/bin/stockfish"  # Path to Stockfish binary ver 18+ (adjust as needed)
STOCKFISH_ANALYSIS_TIME = 0.1  # Analysis time in seconds (adjust for deeper search)

# Configuration: Path to Stockfish executable (verified via Homebrew install)
# installed stockfish via pip and homegrew
# brew install stockfish
# path - /opt/homebrew/Cellar/stockfish/18/bin/stockfish
# /opt/homebrew/Cellar/stockfish/18: 7 files, 113.9MB

# Color definitions for board and UI
# Board theme 1 (default)
LIGHT_SQUARE = (233, 214, 176)
DARK_SQUARE = (164, 128, 92)

# Alternative themes (uncomment to use)
# LIGHT_SQUARE = (240, 217, 181)
# DARK_SQUARE = (181, 136, 99)
# LIGHT_SQUARE = (231, 233, 204)
# DARK_SQUARE = (112, 137, 81)
# LIGHT_SQUARE = (208, 252, 208)
# DARK_SQUARE = (101, 181, 147)

HIGHLIGHT_SELECTED = (255, 255, 0, 100)  # Yellow highlight for selected square
HIGHLIGHT_LEGAL = (0, 255, 0, 100)      # Green highlight for legal moves
TEXT_COLOR = (22, 22, 22)  # Text color
BUTTON_COLOR = (150, 150, 150)  # Button background
BUTTON_HOVER = (200, 200, 200)  # Button hover color

# Piece values for minimax evaluation (fallback when Stockfish is disabled)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.25,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.25,
    chess.QUEEN: 9.75,
    chess.KING: 10000  # High value to prioritize king safety
}

# Piece square tables for positional evaluation (simplified, for white; mirrored for black)
# These tables assign bonuses/penalties based on piece position
PAWN_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
]

KNIGHT_TABLE = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

BISHOP_TABLE = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

ROOK_TABLE = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [5, 10, 10, 10, 10, 10, 10, 5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [-5, 0, 0, 0, 0, 0, 0, -5],
    [0, 0, 0, 5, 5, 0, 0, 0]
]

QUEEN_TABLE = [
    [-20, -10, -10, -5, -5, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 5, 5, 5, 0, -10],
    [-5, 0, 5, 5, 5, 5, 0, -5],
    [0, 0, 5, 5, 5, 5, 0, -5],
    [-10, 5, 5, 5, 5, 5, 0, -10],
    [-10, 0, 5, 0, 0, 0, 0, -10],
    [-20, -10, -10, -5, -5, -10, -10, -20]
]

KING_TABLE = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [20, 20, 0, 0, 0, 0, 20, 20],
    [20, 30, 10, 0, 0, 10, 30, 20]
]

PIECE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_TABLE
}

# Helper function to calculate positional score for a piece
def get_position_score(piece_type, square, color):
    table = PIECE_TABLES[piece_type]
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if color == chess.BLACK:
        rank = 7 - rank  # Mirror for black
    return table[rank][file] / 100.0  # Scaled value

# Evaluation functions for minimax (material, positional, advanced)
def evaluate_material(board):
    score = 0
    bishops_w = bishops_b = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                score += val
                if piece.piece_type == chess.BISHOP: bishops_w += 1
            else:
                score -= val
                if piece.piece_type == chess.BISHOP: bishops_b += 1
    # Bishop pair bonus
    if bishops_w >= 2: score += 0.5
    if bishops_b >= 2: score -= 0.5
    logging.debug(f"Material score: {score}")
    return score

def evaluate_positional(board):
    score = evaluate_material(board)
    pos_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            pos = get_position_score(piece.piece_type, square, piece.color)
            if piece.color == chess.WHITE:
                pos_score += pos
            else:
                pos_score -= pos
    logging.debug(f"Positional bonus: {pos_score}")
    score += pos_score
    return score

def evaluate_advanced(board):
    score = evaluate_positional(board)
    
    # Mobility score
    mobility_score = 0
    board.turn = chess.WHITE
    mobility_white = len(list(board.legal_moves))
    board.turn = chess.BLACK
    mobility_black = len(list(board.legal_moves))
    mobility_score = (mobility_white - mobility_black) * 0.1
    logging.debug(f"Mobility: White {mobility_white}, Black {mobility_black}, Score {mobility_score}")
    score += mobility_score
    
    # King safety (simple castling check)
    king_safety = 0
    if not board.has_kingside_castling_rights(chess.WHITE) and not board.has_queenside_castling_rights(chess.WHITE):
        king_safety += 0.5  # Assume castled if no rights left
    if not board.has_kingside_castling_rights(chess.BLACK) and not board.has_queenside_castling_rights(chess.BLACK):
        king_safety -= 0.5
    logging.debug(f"King safety: {king_safety}")
    score += king_safety
    
    # Pawn structure (penalize doubled pawns)
    pawn_struct = 0
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        files = [0] * 8
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                files[chess.square_file(square)] += 1
        for f in files:
            if f > 1:
                pawn_struct += sign * (1 - f) * 0.5  # Penalty for extra pawns on file
    logging.debug(f"Pawn structure: {pawn_struct}")
    score += pawn_struct
    
    return score

# AI difficulty configurations for minimax
DIFFICULTY_CONFIG = {
    'easy': {'depth': 2, 'use_ab': False, 'eval_func': evaluate_material},
    'medium': {'depth': 4, 'use_ab': True, 'eval_func': evaluate_positional},
    'hard': {'depth': 6, 'use_ab': True, 'eval_func': evaluate_advanced}
}

# Minimax algorithm with optional alpha-beta pruning
def minimax(board, depth, alpha, beta, maximizing, eval_func, use_ab, nodes):
    nodes[0] += 1
    if depth == 0 or board.is_game_over():
        return eval_func(board)
    
    if maximizing:
        max_eval = -inf
        for move in list(board.legal_moves):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, eval_func, use_ab, nodes)
            board.pop()
            max_eval = max(max_eval, eval)
            if use_ab:
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = inf
        for move in list(board.legal_moves):
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, eval_func, use_ab, nodes)
            board.pop()
            min_eval = min(min_eval, eval)
            if use_ab:
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

# Function to get AI move (prefers Stockfish if available, falls back to minimax)
def get_ai_move(board, color, difficulty, stockfish=None):
    start_time = time.time()
    
    if USE_STOCKFISH and stockfish is not None:
        logging.info(f"Stockfish → Analyzing position ({difficulty.upper()} difficulty)")
        logging.debug(f"FEN: {board.fen()}")
        
        stockfish.set_fen_position(board.fen())
        
        # Map difficulty to Stockfish skill level
        skill_map = {'easy': 8, 'medium': 14, 'hard': 20}
        stockfish.set_skill_level(skill_map.get(difficulty, 15))
        
        best_move_uci = stockfish.get_best_move_time(int(STOCKFISH_ANALYSIS_TIME * 1000))
        
        if best_move_uci:
            move = chess.Move.from_uci(best_move_uci)
            evaluation = stockfish.get_evaluation()
            score = evaluation['value'] / 100.0 if evaluation['type'] == 'cp' else evaluation['value']
            
            time_taken = time.time() - start_time
            
            logging.info(f"Stockfish Best Move: {board.san(move)} | "
                         f"Score: {score:+.2f} | "
                         f"Time: {time_taken:.3f}s")
            
            return move, score
    
    # Fallback to minimax if Stockfish fails or disabled
    logging.warning("Stockfish unavailable - falling back to minimax")
    config = DIFFICULTY_CONFIG[difficulty]
    depth = config['depth']
    use_ab = config['use_ab']
    eval_func = config['eval_func']
    
    nodes = [0]
    best_move = None
    best_score = -inf if color == chess.WHITE else inf
    alpha = -inf
    beta = inf
    
    maximizing = (color == chess.WHITE)
    for move in list(board.legal_moves):
        if move.promotion:  # AI promotes to queen by default
            move.promotion = chess.QUEEN
        board.push(move)
        score = minimax(board, depth - 1, alpha, beta, not maximizing, eval_func, use_ab, nodes)
        board.pop()
        logging.debug(f"Considering move {board.san(move)}, score: {score}")
        
        if maximizing:
            if score > best_score:
                best_score = score
                best_move = move
            if use_ab:
                alpha = max(alpha, score)
        else:
            if score < best_score:
                best_score = score
                best_move = move
            if use_ab:
                beta = min(beta, score)
    
    time_taken = time.time() - start_time
    logging.info(f"AI ({difficulty}) best move: {board.san(best_move)}, score: {best_score}, depth: {depth}, nodes: {nodes[0]}, time: {time_taken}s")
    logging.info(f"Nodes per second: {nodes[0] / time_taken if time_taken > 0 else 0}")
    
    return best_move, best_score

# Main ChessGame class
class ChessGame:
    def __init__(self, args):
        pygame.init()  # Initialize Pygame
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))  # Set up display
        pygame.display.set_caption("Chess Game")  # Window title
        self.clock = pygame.time.Clock()  # Clock for FPS control
        self.font = pygame.font.SysFont(None, 24)  # Font for text rendering
        
        # Load piece images (assumed in 'pieces' directory)
        self.pieces_images = {}
        for color in ['white', 'black']:
            for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
                symbol = (piece[0].upper() if piece != 'knight' else 'N') if color == 'white' else (piece[0].lower() if piece != 'knight' else 'n')
                img_path = f'pieces/{color}_{piece}.png'
                if os.path.exists(img_path):
                    img = pygame.image.load(img_path)
                    self.pieces_images[chess.Piece.from_symbol(symbol)] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        
        self.board = chess.Board()  # Initialize chess board
        self.selected_square = None  # Currently selected square
        self.legal_moves = []  # List of legal moves from selected square
        self.move_history = []  # List of moves in SAN notation
        self.game_over = False  # Flag for game end
        self.status = ""  # Status message (e.g., checkmate)
        
        # Button rectangles for UI
        self.new_game_rect = pygame.Rect(BOARD_SIZE + 10, 500, SIDE_PANEL_WIDTH - 20, 30)
        self.quit_rect = pygame.Rect(BOARD_SIZE + 10, 540, SIDE_PANEL_WIDTH - 20, 30)
        
        # PGN-related attributes
        self.start_time = None
        self.white_player = None
        self.black_player = None
        self.result = None
        self.pgn_comments = []  # Comments for each move (e.g., eval scores)
        
        # Command-line arguments
        self.args = args
        
        # Validate arguments
        self.validate_args()
        
        # Select game mode
        self.start_time = time.time()
        self.select_mode()
        self.setup_logging()
        self.stockfish = None
        self.init_stockfish()

    def validate_args(self):
        """Validate command-line arguments for consistency"""
        if self.args.mode not in ['1', '2', '3', None]:
            sys.exit("Invalid --mode: must be 1, 2, or 3")
        
        if self.args.mode == '1':
            if self.args.color:
                sys.exit("--color is not applicable for mode 1 (hvh)")
            if self.args.level:
                sys.exit("--level is not applicable for mode 1 (hvh)")
        
        if self.args.mode == '2':
            if self.args.color and self.args.color not in ['w', 'b']:
                sys.exit("Invalid --color: must be 'w' or 'b'")
            if self.args.level and self.args.level not in ['easy', 'medium', 'hard']:
                sys.exit("Invalid --level: must be 'easy', 'medium', or 'hard'")
        
        if self.args.mode == '3':
            if self.args.color:
                sys.exit("--color is not applicable for mode 3 (aivai)")
            if self.args.level and self.args.level not in ['easy', 'medium', 'hard']:
                sys.exit("Invalid --level: must be 'easy', 'medium', or 'hard'")

    def select_mode(self):
        """Select game mode (Human vs Human, Human vs AI, AI vs AI)"""
        mode = self.args.mode if self.args.mode else input("Select mode:\n1. Human vs Human\n2. Human vs AI\n3. AI vs AI\nEnter choice (1/2/3): ")
        
        if mode == '1':
            self.mode = 'hvh'
            self.difficulty_white = None
            self.difficulty_black = None
            self.human_color = None
            self.white_player = "Human"
            self.black_player = "Human"
        elif mode == '2':
            self.mode = 'hvai'
            side = self.args.color if self.args.color else input("Play as White (w) or Black (b)? ")
            self.human_color = chess.WHITE if side.lower() == 'w' else chess.BLACK
            diff = self.args.level if self.args.level else input("Difficulty (easy/medium/hard): ").lower()
            self.difficulty_white = diff if self.human_color == chess.BLACK else diff
            self.difficulty_black = diff if self.human_color == chess.WHITE else diff
            if self.human_color == chess.WHITE:
                self.white_player = "Human"
                self.black_player = f"AI ({self.difficulty_black})"
            else:
                self.white_player = f"AI ({self.difficulty_white})"
                self.black_player = "Human"
        elif mode == '3':
            self.mode = 'aivai'
            self.human_color = None
            if self.args.level:
                diff_white = self.args.level
                diff_black = self.args.level
            else:
                diff_white = input("White AI difficulty (easy/medium/hard): ").lower()
                diff_black = input("Black AI difficulty (easy/medium/hard): ").lower()
            self.difficulty_white = diff_white
            self.difficulty_black = diff_black
            self.white_player = f"AI ({self.difficulty_white})"
            self.black_player = f"AI ({self.difficulty_black})"
        else:
            print("Invalid, default to Human vs Human")
            self.mode = 'hvh'
            self.white_player = "Human"
            self.black_player = "Human"
        
        logging.info(f"Mode selected: {self.mode}, Difficulties: W-{self.difficulty_white}, B-{self.difficulty_black}, Human color: {self.human_color}")

    def is_human_turn(self):
        """Check if it's a human player's turn"""
        if self.mode == 'hvh':
            return True
        elif self.mode == 'hvai':
            return (self.board.turn == self.human_color)
        elif self.mode == 'aivai':
            return False
        return False

    def setup_logging(self):
        """Set up logging to console and file"""
        root_logger = logging.getLogger('')
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        root_logger.addHandler(console)
        
        if GENERATE_LOG:
            self.timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.start_time))
            log_file = f"chess_game_{self.timestamp}.log"
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            root_logger.setLevel(logging.DEBUG)
            logging.info(f"Logging to file: {log_file}")
        else:
            root_logger.setLevel(logging.INFO)

    def init_stockfish(self):
        """Initialize Stockfish engine if enabled"""
        if not USE_STOCKFISH:
            logging.info("Stockfish is DISABLED - using minimax fallback")
            self.stockfish = None
            return

        logging.info(f"Initializing Stockfish Engine")
        logging.info(f"Binary path: {STOCKFISH_PATH}")
        
        try:
            self.stockfish = Stockfish(
                path=STOCKFISH_PATH,
                depth=18,
                parameters={"Threads": 4, "Hash": 256}
            )
            self.stockfish.set_skill_level(20)
            
            logging.info(f"Stockfish initialized successfully!")
            try:
                major_version = self.stockfish.get_stockfish_major_version()
            except AttributeError:
                major_version = "Unknown"
            logging.info(f"Major version: {major_version}")
            logging.info(f"Skill level set to 20 (maximum)")
            logging.info("Stockfish ready for analysis")
            
        except Exception as e:
            logging.error(f"Failed to initialize Stockfish: {e}")
            logging.warning("Falling back to pure Python minimax")
            self.stockfish = None

    def draw_board(self):
        """Draw the chess board squares"""
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_pieces(self):
        """Draw pieces on the board"""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                img = self.pieces_images.get(piece)
                if img:
                    col = chess.square_file(square)
                    row = 7 - chess.square_rank(square)
                    self.screen.blit(img, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def draw_highlights(self):
        """Highlight selected square and legal moves"""
        if self.selected_square is not None:
            col = chess.square_file(self.selected_square)
            row = 7 - chess.square_rank(self.selected_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(100)
            s.fill((255, 255, 0))
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
        
        for move in self.legal_moves:
            col = chess.square_file(move.to_square)
            row = 7 - chess.square_rank(move.to_square)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            s.set_alpha(100)
            s.fill((0, 255, 0))
            self.screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def draw_side_panel(self):
        """Draw side panel with turn, status, history, and captured material"""
        pygame.draw.rect(self.screen, (200, 200, 200), (BOARD_SIZE, 0, SIDE_PANEL_WIDTH, SCREEN_WIDTH))
        
        turn_text = self.font.render(f"Turn: {'White' if self.board.turn else 'Black'}", True, TEXT_COLOR)
        self.screen.blit(turn_text, (BOARD_SIZE + 10, 10))
        
        status_text = self.font.render(self.status, True, TEXT_COLOR)
        self.screen.blit(status_text, (BOARD_SIZE + 10, 40))
        
        history_text = self.font.render("Move History:", True, TEXT_COLOR)
        self.screen.blit(history_text, (BOARD_SIZE + 10, 70))
        for i, move in enumerate(self.move_history[-10:]):
            move_text = self.font.render(move, True, TEXT_COLOR)
            self.screen.blit(move_text, (BOARD_SIZE + 10, 100 + i * 20))
        
        initial_material = 1*8 + 3.25*2 + 3.25*2 + 5.25*2 + 9.75
        captured_white = initial_material - sum(PIECE_VALUES[p.piece_type] for s in chess.SQUARES if (p := self.board.piece_at(s)) and p.color == chess.WHITE)
        captured_black = initial_material - sum(PIECE_VALUES[p.piece_type] for s in chess.SQUARES if (p := self.board.piece_at(s)) and p.color == chess.BLACK)
        cap_w = self.font.render(f"White captured: {captured_white:.2f}", True, TEXT_COLOR)
        cap_b = self.font.render(f"Black captured: {captured_black:.2f}", True, TEXT_COLOR)
        self.screen.blit(cap_w, (BOARD_SIZE + 10, 350))
        self.screen.blit(cap_b, (BOARD_SIZE + 10, 380))

    def draw_buttons(self):
        """Draw New Game and Quit buttons"""
        pygame.draw.rect(self.screen, BUTTON_COLOR, self.new_game_rect)
        new_game_text = self.font.render("New Game", True, TEXT_COLOR)
        text_rect = new_game_text.get_rect(center=self.new_game_rect.center)
        self.screen.blit(new_game_text, text_rect)
        
        pygame.draw.rect(self.screen, BUTTON_COLOR, self.quit_rect)
        quit_text = self.font.render("Quit", True, TEXT_COLOR)
        text_rect = quit_text.get_rect(center=self.quit_rect.center)
        self.screen.blit(quit_text, text_rect)

    def handle_promotion(self, move):
        """Handle pawn promotion selection (UI dialog)"""
        promo_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        promo_images = [self.pieces_images[chess.Piece(p, self.board.turn)] for p in promo_pieces]
        selected = None
        promo_rects = []
        for i in range(4):
            rect = pygame.Rect(BOARD_SIZE + 10 + i * 40, 410, 30, 30)
            promo_rects.append(rect)
        while selected is None:
            self.screen.fill((200, 200, 200), (BOARD_SIZE, 400, SIDE_PANEL_WIDTH, 200))
            promo_text = self.font.render("Choose promotion:", True, TEXT_COLOR)
            self.screen.blit(promo_text, (BOARD_SIZE + 10, 380))
            for i, img in enumerate(promo_images):
                if img:
                    self.screen.blit(pygame.transform.scale(img, (30, 30)), promo_rects[i].topleft)
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    for i, rect in enumerate(promo_rects):
                        if rect.collidepoint(pos):
                            selected = promo_pieces[i]
                            break
        return selected

    def make_move(self, move, score=None):
        """Make a move on the board and update history"""
        san = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san)
        logging.info(f"Move: {san} ({'White' if self.board.turn else 'Black'})")
        if self.board.is_check():
            logging.warning("Check!")
        
        comment = f"eval: {score:.2f}" if score is not None else ""
        self.pgn_comments.append(comment)
        
        self.check_game_over()

    def check_game_over(self):
        """Check for game over conditions and set status/result"""
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn else "White"
            self.status = f"Checkmate - {winner} wins"
            self.result = "0-1" if winner == "Black" else "1-0"
            self.game_over = True
            logging.info(self.status)
        elif self.board.is_stalemate():
            self.status = "Stalemate"
            self.result = "1/2-1/2"
            self.game_over = True
            logging.info(self.status)
        elif self.board.is_insufficient_material():
            self.status = "Draw - Insufficient material"
            self.result = "1/2-1/2"
            self.game_over = True
            logging.info(self.status)
        elif self.board.can_claim_fifty_moves():
            self.status = "Draw - 50-move rule"
            self.result = "1/2-1/2"
            self.game_over = True
            logging.info(self.status)
        elif self.board.can_claim_threefold_repetition():
            self.status = "Draw - Threefold repetition"
            self.result = "1/2-1/2"
            self.game_over = True
            logging.info(self.status)
        if self.game_over:
            total_moves = len(self.move_history)
            logging.info(f"Game over. Total moves: {total_moves}")
            if GENERATE_PGN:
                self.write_pgn()

    def write_pgn(self):
        """Write game to PGN file"""
        pgn_file = f"chess_game_{self.timestamp}.pgn"
        date = time.strftime("%Y.%m.%d", time.localtime(self.start_time))
        
        headers = f"""[Event "Chess Game"]
[Site "Local"]
[Date "{date}"]
[Round "-"]
[White "{self.white_player}"]
[Black "{self.black_player}"]
[Result "{self.result or '*'}"]

"""
        
        pgn_moves = []
        for i, san in enumerate(self.move_history, 1):
            move_str = san
            comment = self.pgn_comments[i-1] if i-1 < len(self.pgn_comments) else ""
            if comment:
                move_str += f" {{{comment}}}"
            if i % 2 == 1:
                pgn_moves.append(f"{(i + 1) // 2}. {move_str}")
            else:
                pgn_moves[-1] += f" {move_str}"
        
        moves_str = ' '.join(pgn_moves) + f" {self.result or '*'}"
        
        with open(pgn_file, 'w') as f:
            f.write(headers + moves_str + "\n")
        
        logging.info(f"PGN file generated: {pgn_file}")
        self.validate_pgn(pgn_file)  # Validate after writing

    def validate_pgn(self, pgn_file):
        """Validate PGN by parsing and replaying moves"""
        if not os.path.exists(pgn_file):
            logging.error(f"PGN file not found: {pgn_file}")
            return

        try:
            with open(pgn_file, 'r') as f:
                pgn_content = f.read()

            game = chess.pgn.read_game(io.StringIO(pgn_content))
            if game is None:
                raise ValueError("Failed to parse PGN header/moves")

            board = game.board()
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal move in PGN: {board.san(move)}")
                board.push(move)

            logging.info(f"PGN validation PASSED → {pgn_file} is a valid chess game")
        except Exception as e:
            logging.error(f"PGN validation FAILED: {e}")
            logging.error("The generated PGN may be corrupt or contain illegal moves")

    def reset_game(self):
        """Reset game state for a new game"""
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.move_history = []
        self.pgn_comments = []
        self.game_over = False
        self.status = ""
        self.result = None
        self.start_time = time.time()
        self.select_mode()
        self.setup_logging()

    def run(self):
        """Main game loop"""
        running = True
        while running:
            if not self.is_human_turn() and not self.game_over:
                difficulty = self.difficulty_white if self.board.turn else self.difficulty_black
                move, score = get_ai_move(self.board, self.board.turn, difficulty, self.stockfish)
                self.make_move(move, score=score)
                if self.mode == 'aivai':
                    time.sleep(1)  # Delay for AI vs AI visibility
            
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if self.new_game_rect.collidepoint(pos):
                        self.reset_game()
                        continue
                    if self.quit_rect.collidepoint(pos):
                        running = False
                        continue
                    if pos[0] >= BOARD_SIZE:
                        continue
                    if self.is_human_turn() and not self.game_over:
                        col = pos[0] // SQUARE_SIZE
                        row = pos[1] // SQUARE_SIZE
                        square = chess.square(col, 7 - row)
                        
                        if self.selected_square is None:
                            piece = self.board.piece_at(square)
                            if piece and piece.color == self.board.turn:
                                self.selected_square = square
                                self.legal_moves = [m for m in self.board.legal_moves if m.from_square == square]
                        else:
                            for m in self.legal_moves:
                                if m.to_square == square:
                                    move = m
                                    if move.promotion:
                                        promo = self.handle_promotion(move)
                                        move = chess.Move(move.from_square, move.to_square, promo)
                                    self.make_move(move)
                                    break
                            self.selected_square = None
                            self.legal_moves = []
            
            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_side_panel()
            self.draw_buttons()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        # Save PGN if quitting early
        if GENERATE_PGN:
            if self.result is None:
                self.result = "*"
            self.write_pgn()
        
        pygame.quit()

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Entry point of the program.
    # This block ensures the code only runs when this file is executed
    # directly and NOT when imported as a module.
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Set up command-line argument parsing.
    # This allows users to configure the game behavior without modifying code.
    # Example usage:
    #   --mode 2 --color w --level easy
    #   --mode 2 --color b --level medium
    #   --mode 2 --color w --level hard
    # ---------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Chess Game with optional command-line configuration."
    )

    # Game mode selection:
    # 1 = Human vs Human (hvh)
    # 2 = Human vs AI (hvai)
    # 3 = AI vs AI (aivai)
    # We restrict choices to valid options to prevent invalid input.
    parser.add_argument(
        '--mode',
        type=str,
        choices=['1', '2', '3'],
        help='Game mode: 1 (hvh), 2 (hvai), 3 (aivai)'
    )

    # Player color selection (only relevant for Human vs AI mode).
    # 'w' = Human plays White
    # 'b' = Human plays Black
    parser.add_argument(
        '--color',
        type=str,
        choices=['w', 'b'],
        help='For hvai mode: w (white) or b (black)'
    )

    # AI difficulty level configuration.
    # This likely controls depth, evaluation complexity, or search time.
    parser.add_argument(
        '--level',
        type=str,
        choices=['easy', 'medium', 'hard'],
        help='Difficulty level: easy, medium, hard (for AI)'
    )

    # Parse the provided command-line arguments.
    # If invalid arguments are passed, argparse will automatically
    # display an error message and exit the program.
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Ensure required asset directory exists.
    # The 'pieces' folder is expected to contain chess piece images
    # used for rendering the board UI.
    # ---------------------------------------------------------------------
    if not os.path.exists('pieces'):
        # Create the directory if it does not exist.
        # This prevents runtime errors when loading image assets.
        os.makedirs('pieces')

    # Note: we do not automatically populate the 'pieces' directory with images
    # We assume the user manually places the piece image files
    # inside the 'pieces' directory before running the game.
    # You could optionally add validation here to check required files.

    # ---------------------------------------------------------------------
    # Initialize the ChessGame object.
    # Pass the parsed arguments so the game can configure:
    # - Mode (hvh / hvai / aivai)
    # - Player color
    # - AI difficulty level
    # ---------------------------------------------------------------------
    game = ChessGame(args)

    # Start the main game loop.
    # This typically handles:
    # - Event processing (mouse/keyboard)
    # - Move validation
    # - AI computation
    # - Rendering updates
    # - Game state transitions
    game.run()