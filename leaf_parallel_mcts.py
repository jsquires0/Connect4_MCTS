import numpy as np
import random
import connect4
import copy
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import simulate 

# number of parallel rollouts to execute on gpu
LEAF_ROLLOUTS = 128

class MonteCarloTreeSearch():
    """ Implementation of monte carlo tree search for a two
    player game"""

    def __init__(self, root, time, max_rollouts = 400):
        self.root = root
        self.think_time = time
        self.max_rollouts = max_rollouts
        self.c = np.sqrt(2) # UCT exploration param

        self.action, self.rollouts = self.search()

    def selection_expansion(self, node):
        """ 
        Beginning from a root node, repeatedly selects a move and traverses to 
        a corresponding child node. Determining which move to take at each step 
        is handled by the tree policy, which seeks to balance exploration and
        exploitation. 
        
        Selection ends when an encountered node's children are not already
        part of the search tree. At this point,
        one of the missing child nodes is added to the search tree.

        """
        # traverse tree, looking for an unpopulated child node
        while ((len(node.state.valid_moves()) == len(node.children)) and
               (not node.state.terminal)):
            node = self.choose_child(node)

        # return leaf
        if node.state.terminal:
            return node

        # expand tree if not terminal 
        valid_moves = node.state.valid_moves()
        np.random.shuffle(valid_moves)
        for move in valid_moves:
            if move in node.children.keys():
                continue
            else:
                # add a new child node to the tree
                new_child = Node(game_state = connect4.ConnectFour(node.state, 
                                                                   move),  
                                 parent = node)
                node.children[move] = new_child
                break

        return new_child
    
    def simulation(self, node):
        """ 
        Plays the game out from node to a terminal game state according to 
        the rollout policy
        """
        player = node.state.player
        # initialize a node for the rollout
        tmp_node = copy.deepcopy(node)
        tmp_node.player = player % 2 + 1
        while not tmp_node.state.terminal:
            outcome = self.rollout(tmp_node, tmp_node.player)
            tmp_node.player = tmp_node.player % 2 + 1

        tmp_node.state.player = player
        # If node is terminal, return the winner
        return self.rollout(tmp_node, tmp_node.player)

    
    def backup(self, node, outcomes):
        # update win ratios for nodes along selection path
        while node.parent:
            node.increment(outcomes)
            node = node.parent

        self.root.increment(outcomes)
        return

    def choose_child(self, node, use_tree = True):
        """ 
        Choose the best child of the current node according to
        the tree policy, or according to win ratio. Since win ratio corresponds 
        to parent's decision, if this funciton is called with use_tree = False,
        the 'best child' is the node with the lowest win ratio.
        """
        # initialize best score
        if use_tree:
            best = -1
        else:
            best = 2

        winners = []
        # check each child for highest UCT score
        for child in node.children.values():
            # if this function is called during selection, use the tree policy
            if use_tree:
                score = self.tree_policy(node, child)
            # otherwise it's called after backup, use win ratio
            else:
                score = child.wins / child.visits
            
            # compare scores to find highest
            if use_tree:
                if score > best:
                    winners = [child]
                    best = score
                elif score == best:
                    if winners:
                        winners.append(child)
                    else: 
                        winners = [child]

            # compare scores to find lowest
            else:
                if score < best:
                    winners = [child]
                    best = score
                elif score == best:
                    winners.append(child)

        return random.choice(winners)
            
    def tree_policy(self, node, child):
        """
        Upper Confidence Bound applied to Trees (UCT)
        """
        exploit = child.wins/child.visits
        explore = np.sqrt(np.log(node.visits / child.visits))
        
        return exploit + self.c * explore
    
    def rollout(self, node, player):
        if node.state.terminal:
            return node.state.game_over()

        move = self.rollout_policy(node.state.valid_moves())
        outcome = node.state.make_move(move, player)

        return outcome
        
    def rollout_policy(self, choices):
        return random.choice(choices)


    def gpu_simulation(self, node, leaf_rollouts):
        """
        Performs max_rollouts in parallel on gpu, all starting from the passed in
        node.
        """

        # define host inputs
        h_board = node.state.board.astype(np.int32)
        h_occ = node.state.column_occupancies.astype(np.int32)
        h_outcomes = np.zeros(shape = (leaf_rollouts, 1),dtype=np.int32)

        # define device inputs
        d_board = cuda.mem_alloc(h_board.nbytes)
        d_occ = cuda.mem_alloc(h_occ.nbytes)
        d_outcomes = cuda.mem_alloc(h_outcomes.nbytes)
   
        # transfer host -> device
        cuda.memcpy_htod(d_outcomes, h_outcomes)
        cuda.memcpy_htod(d_occ, h_occ)
        cuda.memcpy_htod(d_board,h_board)

        # call kernel
        mod = SourceModule(simulate.gpu_code, no_extern_c = True)
        func = mod.get_function("gpuSimulate")
        player = np.int32(node.state.player)
        func(d_board, d_occ, player, d_outcomes,  block = (leaf_rollouts,1,1))

        # transfer device -> host
        cuda.memcpy_dtoh(h_outcomes, d_outcomes)
        return h_outcomes

    def search(self):
        """ Executes all four stages of the MCTS and chooses a move
        after completing rollouts 
        """
        rollouts = 0
        begin = int(round(time.time()))
        elapsed = begin
        # search until think time is up or max rollouts played
        while (((elapsed - begin) < self.think_time) and
               (rollouts < self.max_rollouts)):
            child = self.selection_expansion(self.root)
            outcomes = self.gpu_simulation(child, leaf_rollouts = LEAF_ROLLOUTS)
            self.backup(child, outcomes)
            rollouts += 1
            elapsed = int(round(time.time()))

        # display evaluations
        #for move, child in self.root.children.items():
            #child.state.show_board()
            #print(child.wins / child.visits)
    
        # find and return the child that results in the higest win rate
        best_node = self.choose_child(self.root, use_tree = False)
        for move, child in self.root.children.items():
            if child == best_node:
                best_move = move

        return best_move, rollouts*LEAF_ROLLOUTS

class Node():
    def __init__(self, game_state = None, parent = None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.state = game_state
        self.wins = 0
    
    def increment(self, outcomes):
        """ 
        Update a node's statistics during backup. Since turns alternate, and 
        action decisions are based on child node stats, win total is updated
        only if the opposing player wins.
        """
        self.visits += outcomes.shape[0]
        self.wins += outcomes.sum()

        return
