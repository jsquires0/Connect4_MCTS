import numpy as np
import random
import connect4
import copy


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import simulate 
class MonteCarloTreeSearch():
    """ Implementation of monte carlo tree search for a two
    player game"""

    def __init__(self, root, n_rollouts = 50):
        self.root = root
        self.n_rollouts = n_rollouts
        self.c = np.sqrt(2) # UCT exploration param

        self.action = self.search()

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

    
    def backup(self, node, outcome):
        # update win ratios for nodes along selection path
        while node.parent:
            node.increment(outcome)
            node = node.parent

        self.root.increment(outcome)
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

    def search(self):
        """ Executes all four stages of the MCTS and chooses a move
        after completing rollouts 
        """
        rollouts = 0
        while rollouts < self.n_rollouts:
            child = self.selection_expansion(self.root)
            
            # test pycuda call
            a = np.random.randn(4,4)
            a = a.astype(np.float32)
            a_gpu = cuda.mem_alloc(a.nbytes)
            cuda.memcpy_htod(a_gpu,a)
            mod = SourceModule(simulate.kernel_c_code);
            func = mod.get_function("doublify")
            func(a_gpu, block = (4,4,1))
            a_doubled = np.empty_like(a)
            cuda.memcpy_dtoh(a_doubled, a_gpu)
            print(a_doubled)
            print(a)
            
            outcome = self.simulation(child)
            self.backup(child, outcome)
            rollouts += 1

        # display evaluations
        # for move, child in self.root.children.items():
            #child.state.show_board()
            #print(child.wins, child.visits)
    
        # find and return the child that results in the higest win rate
        best_node = self.choose_child(self.root, use_tree = False)
        for move, child in self.root.children.items():
            if child == best_node:
                best_move = move

        return best_move

class Node():
    def __init__(self, game_state = None, parent = None):
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.state = game_state
        self.wins = 0
    
    def increment(self, winner):
        """ 
        Update a node's statistics during backup. Since turns alternate, and 
        action decisions are based on child node stats, win total is updated
        only if the opposing player wins.
        """
        self.visits += 1

        # if not a draw
        if (winner != -1):
            if self.state.player != winner:
                self.wins += 1
            elif self.state.player == winner:
                self.wins -= 1

        return
