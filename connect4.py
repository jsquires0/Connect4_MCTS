import numpy as np 

# define board dimensions
ROWS = 6
COLS = 7
unicode_chars = {0:u'\u25AB', 1:u'\u25CE', 2:u'\u25C9' } #F, B

class ConnectFour():
    """ 
    Contains functions needed to track the current
    Connect 4 game state and available actions
    """

    def __init__(self):
        """ Initializes an empty game board"""
        self._reset()

    def _reset(self):
        self.board = np.zeros(shape = (ROWS, COLS), dtype = np.int8)
        self.column_occupancies = np.zeros(shape = COLS, dtype = np.int8)
        self.move_count = 0
        self.terminal = False

    def valid_moves(self):
        """ return column indexes where pieces may be dropped"""
        return np.where(self.column_occupancies < ROWS)[0]

    def make_move(self, col, checker):
        """
        Places a checker in a column and increments move counter

        Parameters
        ----------
        col
            index of column to place checker
        checker
            1 or 2, representing which player is placing the checker
        """
        row = self.column_occupancies[col]
        self.board[row][col] = checker
        self.column_occupancies[col] += 1
        self.move_count += 1

        return

    def draw(self):
        # could also use valid_moves()
        return self.move_count == ROWS*COLS

    def win(self):
        """
        Checks if either player has won the game

        Returns
        -----------
        0: No win, 1: Player 1 win, 2: Player 2 win
        """
        # check for horizontal or vertical win
        winner = self.four_row_col()
        if winner:
            return winner
        else:
            # check for diagonal win
            winner = self.four_diagonal()
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

    def take_human_turn(self, player):
        """ Prompt user to place a checker """
        print('Choose a column (0-6) \n')
        col = -1
        while col not in self.valid_moves():
            col = int(input('>>'))
        self.make_move(col, player)
        self.last_player = player

        return

    def take_MCTS_turn(self):
        return

    def play(self):
        # player 1 starts
        self.last_player = 2
        
        while not (self.win() or self.draw()):
            player = (1 if self.last_player == 2 else 2)
            self.take_human_turn(player)
            self.show_board()
            
        print('Game over')
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
    C4 = ConnectFour()
    C4.play()
