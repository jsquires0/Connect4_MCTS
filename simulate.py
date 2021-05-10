gpu_code = """
#include <curand.h>
#include <curand_kernel.h>
extern "C"
{
    #define COLS 7 
    #define ROWS 6

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
        //*arr = (int *)realloc(*arr, numValidMoves * sizeof(int));
        *arr = validMoves;

        return numValidMoves;
    }


    __device__ int getOutcome(int boardState[ROWS][COLS], int numValidMoves)
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

        // check for a draw 
        if(!numValidMoves)
        {
           return -1; 
        }

        // game is not over
        return 0;
    }

    __global__ void gpuSimulate(int *boardState, 
                             int *occupancies,
                             int player,
                             int *results)
    {
        int idx = threadIdx.x + threadIdx.y*ROWS; 
        int winner = -99;

        // populate a copy of the input board
        int tmpBoard[ROWS][COLS];
        for (int r = 0; r < ROWS; r++)
        {
            for (int c = 0; c < COLS; c++) 
            {
                tmpBoard[r][c] = boardState[r + c*COLS];
            }
        }

        // populate a copy of the col occupancies
        int tmpOccupancies[COLS];
        for (int c = 0; c < COLS; c++)
        {
            tmpOccupancies[c] = occupancies[c];
        }

        // current node's player
        int tmpPlayer = player % 2 + 1;
        
        // make valid moves until board is in a terminal state
        int nonTerminal = 1;
        //TODO check if game is over. if yes, return winner
        
        while (nonTerminal)
        {
            //make a move from the list of valid moves
            int numValidMoves;
            int *validMoves;
            numValidMoves = getValidMoves(tmpOccupancies, &validMoves);

            // choose a random move
            curandState_t state;
            curand_init((unsigned long long)clock(), idx, 0, &state);
            int n = curand(&state) % (numValidMoves + 1);
            int col = validMoves[n];

            // make the move
            int row = tmpOccupancies[col];
            tmpBoard[row][col] = tmpPlayer;
            tmpOccupancies[col] ++;

            // check for winner
            winner = getOutcome(tmpBoard, numValidMoves);
            if (winner != 0)
            {
                nonTerminal = 0;
            }
            //update the player
            tmpPlayer = tmpPlayer % 2 + 1;
        }
        
        // store outcome
        if (winner == player){
            results[idx] = -1;
        }
        else if (!winner)
        {
            results[idx] = 0;
        }
        else
        {
            results[idx] = 1;
        }
    }
}
"""
