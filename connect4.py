import numpy as np 
import argparse
from copy import deepcopy as copy

# define board dimensions
ROWS = 6
COLS = 7
# define length of time for MCTS to choose a move (sec)
THINK_DURATION = 120

unicode_chars = {0:u'\u25AB', 1:u'\u25CE', 2:u'\u25C9' }

class ConnectFour():
    """ 
    Contains functions needed to track the current
    Connect 4 game state and available actions
    """

    def __init__(self, last_state = None, move = None):
        """ Initialize a game state """
        self.last_state = last_state
        self.move = move
        self._reset(move)   
        
    def _reset(self, move):
        
        if self.last_state:
            # copy previous board, column occupancies
            self.board = np.copy(self.last_state.board)
            self.column_occupancies = np.copy(self.last_state.column_occupancies)
            self.terminal = False
            self.player = copy(self.last_state.player)
            # increment the previous state
            self.make_move(self.move, self.player)

        else:
            # initialize an empty game board
            self.board = np.zeros(shape = (ROWS, COLS), dtype = np.int8)
            self.column_occupancies = np.zeros(shape = COLS, dtype = np.int8)
            self.terminal = False
            # player 1 moves on even counts, 2 on odd
            self.player = 1
    
    def valid_moves(self):
        """ return column indexes where pieces may be dropped"""
        return np.where(self.column_occupancies < ROWS)[0]

    def make_move(self, col, player):
        """
        Places a checker in a column and increments move counter

        Parameters
        ----------
        col
            index of column to place checker
        player
            1 or 2, representing which player is placing the checker
        """
        row = self.column_occupancies[col]
        self.board[row][col] = player
        self.column_occupancies[col] += 1

        outcome =  self.game_over()
        return outcome

    def game_over(self):
        """ Checks if the game is over. 

        Returns
        -----------
        -1: Draw, 0: No win, 1: Player 1 win, 2: Player 2 win
        """
        winner = self.win()
        
        if self.draw() and not winner:
            return -1

        return winner

    def draw(self):
        # could also use valid_moves()
        self.terminal = (self.valid_moves().size == 0)
        return self.terminal

    def win(self):
        """
        Checks if either player has won the game

        Returns
        -----------
        0: No win, 1: Player 1 win, 2: Player 2 win
        """
        # check for horizontal or vertical win
        winner = self.four_row_col() or self.four_diagonal()
        self.terminal = bool(winner)

        return winner

    def four_row_col(self):
        """ 
        Checks rows and cols for four sequential checkers 
        
        Returns
        -----------
        0: No win, 1: Player 1 win, 2: Player 2 win
        """

        # row win
        for j in range(COLS-3):
            for i in range(ROWS):
                if ((self.board[i][j] == 
                     self.board[i][j+1] == 
                     self.board[i][j+2] == 
                     self.board[i][j+3]) and
                     self.board[i][j]):

                    return self.board[i][j]
        # col win
        for j in range(COLS):
            for i in range(ROWS-3):
                if ((self.board[i][j] == 
                     self.board[i+1][j] ==  
                     self.board[i+2][j] ==  
                     self.board[i+3][j]) and
                     self.board[i][j]):

                    return self.board[i][j]
        return 0

    def four_diagonal(self):
        """ 
        Check diagonals for four sequential checkers 
        
        Returns
        -----------
        0: No win, 1: Player 1 win, 2: Player 2 win
        """

        for j in range(COLS-3):
            for i in range(ROWS-3):
                if ((self.board[i][j] ==  
                     self.board[i+1][j+1] ==  
                     self.board[i+2][j+2] == 
                     self.board[i+3][j+3]) and
                     self.board[i][j]):

                    return self.board[i][j]

        for j in range(COLS-3):
            for i in range(3, ROWS):
                if ((self.board[i][j] == 
                     self.board[i-1][j+1] ==  
                     self.board[i-2][j+2] ==  
                     self.board[i-3][j+3]) and
                     self.board[i][j]):

                    return self.board[i][j]
        return 0

    def take_human_turn(self, player = 1):
        """ Prompt user to place a checker """
        col_id = COLS - 1
        checker = self.encode_position(player)
        col = -1
        while col not in self.valid_moves():
            col = int(input(f'Choose a column [0-{col_id}] to place {checker}: '))
        outcome = self.make_move(col, player)
        self.show_board()
        return outcome

    def take_MCTS_turn(self, player = 2):
        node = mcts.Node(self, parent = None)
        mcts_ai = mcts.MonteCarloTreeSearch(node, THINK_DURATION)
        return mcts_ai.action, mcts_ai.rollouts

    def play_human_vs_AI(self):
        """ Plays a complete game against the MCTS AI. Human player
        always starts
        """

        while not (self.win() or self.draw()):
            # human moves
            if self.player == 1:
                outcome = self.take_human_turn(self.player)
            # AI moves
            else:
                print('MCTS thinking..')
                #outcome = self.take_human_turn(self.player)
                mcts_move, rollouts = self.take_MCTS_turn(self.player)
                print(f'MCTS executed {rollouts} rollouts')
                outcome = self.make_move(mcts_move, self.player)
                self.show_board()

            self.player = self.player % 2 + 1
        win_dict = {1: 'Human', 2: 'MCTS AI'}
        print(f'Game over - {win_dict[outcome]} wins')
        return

    def show_board(self):
        """
        Visualizes the current game state
        """
        board = u'\n'
        for i in range(ROWS):
            board +=  u'|' + ' '
            for j in range(COLS):
                # flip the board since game progresses bottom to top
                c = self.encode_position(np.flipud(self.board)[i,j])
                board += c + ' '
            board += u'|' + '\n'
        print(board)
        return

    def encode_position(self, loc):
        """
        Each location in the board may be represented by either a 0 (unoccupied)
        a 1 (player 1's checker) or a 2 (player 2's checker).
        This function translates these integers into unicode geometric shapes
        for easy interpretation

        Parameters
        -----------
        loc
            An integer (0, 1, or 2) representing the the state of a particular
            location on the board
        Returns
        ----------
        Unicode string representing a geometric shape
        """
        return unicode_chars[loc]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.device == 'cpu':
        import serial_mcts as mcts
    else:
        import leaf_parallel_mcts as mcts

    C4 = ConnectFour()
    C4.play_human_vs_AI()
