
gpu_code = """
extern "C"
{
    #define COLS 7 
    #define ROWS 6
    __device__ int getValidMoves(int *occupancies)
    {
        //make a move from the list of valid moves
        int numValidMoves = 0;
        for(int c = 0; c < COLS; c++)
        {
            if (occupancies[c] < ROWS)
            {
                numValidMoves += 1;
            }
        }
        int validMoves[numValidMoves];
        for(int c = 0; c < COLS; c++)
        {
            if (occupancies[c] < ROWS)
            {
                validMoves[c] = c;
            }
        }
    }
    __device__ int getOutcome(int *boardState, int *validMoves)
    {
        
    }

    __global__ void doublify(int *boardState, 
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
        //TODO check if game is over
        
        while (nonTerminal)
        {

            //make a move from the list of valid moves
            int numValidMoves = 0;
            for(int c = 0; c < COLS; c++)
            {
                if (tmpOccupancies[c] < ROWS)
                {
                    numValidMoves += 1;
                }
            }
            int validMoves[numValidMoves];
            for(int c = 0; c < COLS; c++)
            {
                if (tmpOccupancies[c] < ROWS)
                {
                    validMoves[c] = c;
                }
            }
            // choose a move
            int n = rand()%(numValidMoves+1);
            int move = validMoves[n];

            // make the move (update board, occ, check for winner
            int row = tmpOccupancies[move]
            tmpBoard[row][move] = tmpPlayer;
            // check for winner
            
            //update the player
            tmpPlayer = tmpPlayer % 2 + 1;
        }

        tmpPlayer = player;
        
        


        // return winner
        // call game over with original boardState, 
        //results[idx] = winner;
        boardState[idx] *=2;
    }




}
"""
