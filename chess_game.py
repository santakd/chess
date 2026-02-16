#   __.-._
#   '-._"7'
#    /'.-c
#    |  //
#   _)_/||
#
# chess_game.py - A comprehensive chess game with AI opponents, multiple difficulty levels, and detailed logging.
# Author: santakd
# Contact: santakd at gmail dot com
# Date: February 15, 2026
# Version: 1.0.8
# License: MIT License 

# pip install pygame
# pip install chess

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

# Constants
SCREEN_WIDTH = 800
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
SIDE_PANEL_WIDTH = SCREEN_WIDTH - BOARD_SIZE
FPS = 60
GENERATE_LOG = True  # Generate timestamped log file for normal logging
GENERATE_PGN = True  # Generate PGN file for each game

# Colors
# Board theme 1
LIGHT_SQUARE = (233, 214, 176)
DARK_SQUARE = (164, 128, 92)

# Board theme 2 (uncomment to use)
# LIGHT_SQUARE = (240, 217, 181)
# DARK_SQUARE = (181, 136, 99)

# Board theme 3 (uncomment to use)
# LIGHT_SQUARE = (231, 233, 204)
# DARK_SQUARE = (112, 137, 81)

# Board theme 4 (uncomment to use)
# LIGHT_SQUARE = (208, 252, 208)
# DARK_SQUARE = (101, 181, 147)

HIGHLIGHT_SELECTED = (255, 255, 0, 100)  # Yellow with alpha
HIGHLIGHT_LEGAL = (0, 255, 0, 100)      # Green with alpha
TEXT_COLOR = (0, 0, 0)
BUTTON_COLOR = (150, 150, 150)
BUTTON_HOVER = (200, 200, 200)

# Piece values for material evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3.25,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.25,
    chess.QUEEN: 9.75,
    chess.KING: 10000  # High value to protect king
}

# Piece square tables (simplified, for white; mirror for black)
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

# Helper to get position score
def get_position_score(piece_type, square, color):
    table = PIECE_TABLES[piece_type]
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if color == chess.BLACK:
        rank = 7 - rank
    return table[rank][file] / 100.0  # Scale to reasonable values

# Evaluation functions
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
    # Removed debug logging to reduce repetition
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
    # Removed debug logging to reduce repetition
    return score

def evaluate_advanced(board):
    score = evaluate_positional(board)
    
    # Mobility
    mobility_score = 0
    board.turn = chess.WHITE
    mobility_white = len(list(board.legal_moves))
    board.turn = chess.BLACK
    mobility_black = len(list(board.legal_moves))
    mobility_score = (mobility_white - mobility_black) * 0.1
    # Removed debug logging to reduce repetition
    
    score += mobility_score
    
    # King safety (simple: bonus if castled)
    king_safety = 0
    if board.has_kingside_castling_rights(chess.WHITE) == False and board.has_queenside_castling_rights(chess.WHITE) == False:
        king_safety += 0.5  # Assume castled if no rights left (simplification)
    if board.has_kingside_castling_rights(chess.BLACK) == False and board.has_queenside_castling_rights(chess.BLACK) == False:
        king_safety -= 0.5
    # Removed debug logging to reduce repetition
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
                pawn_struct += sign * (1 - f) * 0.5  # -0.5 per extra pawn
    # Removed debug logging to reduce repetition
    score += pawn_struct
    
    return score

# AI difficulties config
DIFFICULTY_CONFIG = {
    'easy': {'depth': 2, 'use_ab': False, 'eval_func': evaluate_material},
    'medium': {'depth': 4, 'use_ab': True, 'eval_func': evaluate_positional},
    'hard': {'depth': 6, 'use_ab': True, 'eval_func': evaluate_advanced}
}

# Minimax with optional alpha-beta
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

# Get AI move
def get_ai_move(board, color, difficulty):
    config = DIFFICULTY_CONFIG[difficulty]
    depth = config['depth']
    use_ab = config['use_ab']
    eval_func = config['eval_func']
    
    start_time = time.time()
    nodes = [0]
    best_move = None
    best_score = -inf if color == chess.WHITE else inf
    alpha = -inf
    beta = inf
    
    maximizing = (color == chess.WHITE)
    for move in list(board.legal_moves):
        if move.promotion:  # AI always promotes to queen
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
    # Branching factor approx: nodes / (nodes at depth-1), but skip for now
    
    return best_move, best_score

# Class for the game
class ChessGame:
    def __init__(self, args):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
        pygame.display.set_caption("Chess Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        # Load pieces (assume in 'pieces' directory)
        self.pieces_images = {}
        for color in ['white', 'black']:
            for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
                symbol = (piece[0].upper() if piece != 'knight' else 'N') if color == 'white' else (piece[0].lower() if piece != 'knight' else 'n')
                img_path = f'pieces/{color}_{piece}.png'
                if os.path.exists(img_path):
                    img = pygame.image.load(img_path)
                    self.pieces_images[chess.Piece.from_symbol(symbol)] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves = []
        self.move_history = []
        self.game_over = False
        self.status = ""
        
        # Button rects
        self.new_game_rect = pygame.Rect(BOARD_SIZE + 10, 500, SIDE_PANEL_WIDTH - 20, 30)
        self.quit_rect = pygame.Rect(BOARD_SIZE + 10, 540, SIDE_PANEL_WIDTH - 20, 30)
        
        # PGN related
        self.start_time = None
        self.white_player = None
        self.black_player = None
        self.result = None
        self.pgn_comments = []  # List to store comments for each move
        
        # Command-line args
        self.args = args
        
        # Validate command-line arguments
        self.validate_args()
        
        # Mode selection (uses args or prompts)
        # Set start_time before selecting mode and setup_logging, 
        # this also ensures the timestamp in logs and PGN is consistent with game start
    
        self.start_time = time.time()  
        self.select_mode()
        self.setup_logging()
    
    def validate_args(self):
        # Post-parsing validation for argument consistency
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
        # Use command-line args if provided, else prompt
        mode = self.args.mode if self.args.mode else input("Select mode:\n1. Human vs Human\n2. Human vs AI\n3. AI vs AI\nEnter choice (1/2/3): ")
        
        if mode == '1':
            self.mode = 'hvh'
            self.difficulty_white = None
            self.difficulty_black = None
            self.human_color = None  # Both human
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
            # For aivai, if level provided, use for both; else prompt separately
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
        if self.mode == 'hvh':
            return True
        elif self.mode == 'hvai':
            return (self.board.turn == self.human_color)
        elif self.mode == 'aivai':
            return False
        return False
    
    # Set up logging for better traceability and error reporting
    def setup_logging(self):
        # Remove existing handlers to avoid duplicate logs
        root_logger = logging.getLogger('')
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Set up console handler (always, for output)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        root_logger.addHandler(console)
        
        if GENERATE_LOG:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.start_time))
            log_file = f"chess_game_{timestamp}.log"
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)  # Set to INFO to reduce debug spam
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            root_logger.addHandler(fh)
            root_logger.setLevel(logging.DEBUG)  # Overall level DEBUG, but file is INFO
            logging.info(f"Logging to file: {log_file}")
        else:
            root_logger.setLevel(logging.INFO)  # Only console if no file
    
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    
    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                img = self.pieces_images.get(piece)
                if img:
                    col = chess.square_file(square)
                    row = 7 - chess.square_rank(square)
                    self.screen.blit(img, (col * SQUARE_SIZE, row * SQUARE_SIZE))
    
    def draw_highlights(self):
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
        # Background
        pygame.draw.rect(self.screen, (200, 200, 200), (BOARD_SIZE, 0, SIDE_PANEL_WIDTH, SCREEN_WIDTH))
        
        # Turn
        turn_text = self.font.render(f"Turn: {'White' if self.board.turn else 'Black'}", True, TEXT_COLOR)
        self.screen.blit(turn_text, (BOARD_SIZE + 10, 10))
        
        # Status
        status_text = self.font.render(self.status, True, TEXT_COLOR)
        self.screen.blit(status_text, (BOARD_SIZE + 10, 40))
        
        # Move history
        history_text = self.font.render("Move History:", True, TEXT_COLOR)
        self.screen.blit(history_text, (BOARD_SIZE + 10, 70))
        for i, move in enumerate(self.move_history[-10:]):  # Last 10
            move_text = self.font.render(move, True, TEXT_COLOR)
            self.screen.blit(move_text, (BOARD_SIZE + 10, 100 + i * 20))
        
        # Captured (simple count)
        initial_material = 1*8 + 3.25*2 + 3.25*2 + 5.25*2 + 9.75  # Adjusted for floats
        captured_white = initial_material - sum(PIECE_VALUES[p.piece_type] for s in chess.SQUARES if (p := self.board.piece_at(s)) and p.color == chess.WHITE)
        captured_black = initial_material - sum(PIECE_VALUES[p.piece_type] for s in chess.SQUARES if (p := self.board.piece_at(s)) and p.color == chess.BLACK)
        cap_w = self.font.render(f"White captured: {captured_white:.2f}", True, TEXT_COLOR)
        cap_b = self.font.render(f"Black captured: {captured_black:.2f}", True, TEXT_COLOR)
        self.screen.blit(cap_w, (BOARD_SIZE + 10, 350))
        self.screen.blit(cap_b, (BOARD_SIZE + 10, 380))
    
    def draw_buttons(self):
        # New Game button
        pygame.draw.rect(self.screen, BUTTON_COLOR, self.new_game_rect)
        new_game_text = self.font.render("New Game", True, TEXT_COLOR)
        text_rect = new_game_text.get_rect(center=self.new_game_rect.center)
        self.screen.blit(new_game_text, text_rect)
        
        # Quit button
        pygame.draw.rect(self.screen, BUTTON_COLOR, self.quit_rect)
        quit_text = self.font.render("Quit", True, TEXT_COLOR)
        text_rect = quit_text.get_rect(center=self.quit_rect.center)
        self.screen.blit(quit_text, text_rect)
    
    def handle_promotion(self, move):
        # Simple dialog: draw options on side panel
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
        current_player_color = 'White' if self.board.turn else 'Black'
        if self.mode == 'hvh':
            player = 'Human'
        elif self.mode == 'hvai':
            if self.board.turn == self.human_color:
                player = 'Human'
            else:
                player = 'AI'
        elif self.mode == 'aivai':
            player = 'AI'
        else:
            player = 'Unknown'
        
        san = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san)
        logging.info(f"Move: {san} ({current_player_color})")
        if self.board.is_check():
            logging.warning("Check!")
        
        comment = ""
        if score is not None:
            comment = f"eval: {score:.2f}"
        self.pgn_comments.append(comment)
        
        self.check_game_over()
    
    def check_game_over(self):
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
            # Log overall stats
            total_moves = len(self.move_history)
            logging.info(f"Game over. Total moves: {total_moves}")
            if GENERATE_PGN:
                self.write_pgn()
            # Time per player not tracked, skip
    
    def write_pgn(self):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(self.start_time))
        pgn_file = f"chess_game_{timestamp}.pgn"
        date = time.strftime("%Y.%m.%d", time.localtime(self.start_time))
        
        headers = f"""[Event "Chess Game"]
[Site "Local"]
[Date "{date}"]
[Round "-"]
[White "{self.white_player}"]
[Black "{self.black_player}"]
[Result "{self.result}"]

"""
        
        pgn_moves = []
        for i, san in enumerate(self.move_history, 1):
            move_str = san
            comment = self.pgn_comments[i-1]
            if comment:
                move_str += f" {{{comment}}}"
            if i % 2 == 1:
                pgn_moves.append(f"{(i + 1) // 2}. {move_str}")
            else:
                pgn_moves[-1] += f" {move_str}"
        
        moves_str = ' '.join(pgn_moves) + f" {self.result}"
        
        with open(pgn_file, 'w') as f:
            f.write(headers + moves_str + "\n")
        
        logging.info(f"PGN file generated: {pgn_file}")
    
    def reset_game(self):
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
        running = True
        while running:
            if not self.is_human_turn() and not self.game_over:
                difficulty = self.difficulty_white if self.board.turn else self.difficulty_black
                move, score = get_ai_move(self.board, self.board.turn, difficulty)
                self.make_move(move, score=score)
                if self.mode == 'aivai':
                    time.sleep(1)  # Delay for visibility
            
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
        
        pygame.quit()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Chess Game with optional command-line configuration.")
    parser.add_argument('--mode', type=str, choices=['1', '2', '3'], help='Game mode: 1 (hvh), 2 (hvai), 3 (aivai)')
    parser.add_argument('--color', type=str, choices=['w', 'b'], help='For hvai mode: w (white) or b (black)')
    parser.add_argument('--level', type=str, choices=['easy', 'medium', 'hard'], help='Difficulty level: easy, medium, hard (for AI)')
    args = parser.parse_args()
    
    # Create pieces directory if needed (assume images are there)
    if not os.path.exists('pieces'):
        os.makedirs('pieces')
        
    # Assume user places images there
    game = ChessGame(args)
    game.run()