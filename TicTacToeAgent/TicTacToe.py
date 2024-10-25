import numpy as np
import random

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # 3x3 board represented as a list
        self.current_winner = None  # Keep track of the winner!

    def available_moves(self):
        """Return a list of indices for available moves."""
        return [i for i, spot in enumerate(self.board) if spot == ' ']
    
    def print_board(self):
        for row in [self.board[i * 3:(i + 1) * 3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')


    def empty_squares(self):
        """Return a list of indices for empty squares."""
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        """Place a letter on the board at the specified square."""
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def winner(self, square, letter):
        """Check if the current player has won."""
        # Check rows, columns, and diagonals for a win
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([spot == letter for spot in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([spot == letter for spot in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            if all([spot == letter for spot in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            if all([spot == letter for spot in diagonal2]):
                return True
        return False

    def reset(self):
        """Reset the game to its initial state."""
        self.board = [' '] * 9
        self.current_winner = None

    def is_terminal(self):
        """Check if the game has reached a terminal state."""
        if self.current_winner:
            return True
        if not self.available_moves():  # Changed to check for available moves
            return True
        return False
