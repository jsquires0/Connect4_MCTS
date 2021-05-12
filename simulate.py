gpu_code = """
#include <curand.h>
#include <curand_kernel.h>
extern "C"
{
    #define COLS 7 
    #define ROWS 6

    // determines available legal moves in the position
    __device__ int getValidMoves(int *occupancies, int **arr)
    {   
        // initialize array to be length COLS
        *arr = (int *) malloc(COLS * sizeof(int));
        //make a move from the list of valid moves
        int numValidMoves = 0;
        for(int c = 0; c < COLS; c++)
        {
            if (occupancies[c] < ROWS)
            {
                numValidMoves += 1;
            }
        }
        int validMoves[COLS];
        int i = 0;
        // populate array of valid moves
        for(int c = 0; c < COLS; c++)
        {
            if (occupancies[c] < ROWS)
            {
                validMoves[i] = c;
                i++;
            }
        }
        // reallocate array to have size numValidMoves
        *arr = validMoves;

        return numValidMoves;
    }

    // checks board for four vertical or horizontal checkers in a row
    __device__ int fourRowCol(int boardState[ROWS][COLS])
    {
        //row win?
        for (int j=0; j < (COLS-3); j++)
        {
            for (int i=0; i < ROWS; i++)
            {
                if ((boardState[i][j] == boardState[i][j+1]) &&
                    (boardState[i][j+1] == boardState[i][j+2]) &&
                    (boardState[i][j+2] == boardState[i][j+3]) &&
                    (boardState[i][j] != 0))
                    {
                        return boardState[i][j];
                    }
            }
        }
        // col win?
        for (int j=0; j < (COLS); j++)
        {
            for (int i=0; i < (ROWS-3); i++)
            {
                if ((boardState[i][j] == boardState[i+1][j]) &&
                    (boardState[i+1][j] == boardState[i+2][j]) &&
                    (boardState[i+2][j] == boardState[i+3][j]) &&
                    (boardState[i][j] != 0))
                    {
                        return boardState[i][j];
                    }
            }
        }
        // no vertical or horizontal win
        return 0;
    }

    // checks board for four diagonal checkers in a row
    __device__ int fourDiagonal(int boardState[ROWS][COLS])
    {
        // diagonal case 1?
        for (int j=0; j < (COLS-3); j++)
        {
            for (int i=0; i < (ROWS-3); i++)
            {
                if ((boardState[i][j] == boardState[i+1][j+1]) &&
                    (boardState[i+1][j+1] == boardState[i+2][j+2]) &&
                    (boardState[i+2][j+2] == boardState[i+3][j+3]) &&
                    (boardState[i][j] != 0))
                    {
                        return boardState[i][j];
                    }
            }
        }
        // diagonal case 2?
        for (int j=0; j < (COLS-3); j++)
        {
            for (int i=3; i < (ROWS); i++)
            {
                if ((boardState[i][j] == boardState[i-1][j+1]) &&
                    (boardState[i-1][j+1] == boardState[i-2][j+2]) &&
                    (boardState[i-2][j+2] == boardState[i-3][j+3]) &&
                    (boardState[i][j] != 0))
                    {
                        return boardState[i][j];
                    }
            }
        }

        // no diagonal win
        return 0;
    }

    // checks if the game has ended
    __device__ int getOutcome(int boardState[ROWS][COLS], int numValidMoves)
    {
        int winner;
        winner = fourRowCol(boardState);
        if (winner != 0)
        {
            return winner;
        }
        winner = fourDiagonal(boardState);
        if (winner != 0)
        {
            return winner;
        }

        // check for a draw 
        if(!numValidMoves)
        {
           return -1; 
        }

        // game has not ended
        return 0;
    }
    // makes a random move
    __device__ int randomRollout(int *occupancies, int boardState[ROWS][COLS],
                                 int player, int winner, int nonTerminal, int idx)
    {
         //make a move from the list of valid moves
        int numValidMoves;
        int *validMoves;
        numValidMoves = getValidMoves(occupancies, &validMoves);

        // choose a random move
        curandState_t state;
        curand_init((unsigned long long)clock(), idx, 0, &state);
        int n = curand(&state) % (numValidMoves + 1);
        int col = validMoves[n];

        // make the move
        int row = occupancies[col];
        boardState[row][col] = player;
        occupancies[col]++;

        // check for winner
        winner = getOutcome(boardState, numValidMoves);
        if (winner != 0)
        {
            nonTerminal = 0;
        }
        return nonTerminal;
    }
    
    // performs the simulation stage of MCTS
    __global__ void gpuSimulate(int *boardState, 
                             int *occupancies,
                             int player,
                             int *results)
    {
        int idx = threadIdx.x + threadIdx.y*ROWS; 
        int winner = -99;

        // copy input board
        int tmpBoard[ROWS][COLS];
        for (int r = 0; r < ROWS; r++)
        {
            for (int c = 0; c < COLS; c++) 
            {
                tmpBoard[r][c] = boardState[r + c*COLS];
            }
        }

        // copy col occupancies
        int tmpOccupancies[COLS];
        for (int c = 0; c < COLS; c++)
        {
            tmpOccupancies[c] = occupancies[c];
        }

        // current node's player
        int tmpPlayer = player % 2 + 1;
        
        // make valid moves until board is in a terminal state
        int nonTerminal = 1;
        while (nonTerminal)
        {
            // complete a random playout
            nonTerminal = randomRollout(tmpOccupancies, tmpBoard, tmpPlayer, 
                                        winner, nonTerminal,  idx);
            // update the player
            tmpPlayer = tmpPlayer % 2 + 1;
        }
        
        // store outcome
        if (winner == player){results[idx] = -1;}
        else if (!winner){results[idx] = 0;}
        else{results[idx] = 1;}
    }
}
"""
