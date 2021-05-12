import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      return (evaluate(board), [], {})
    else:
      moveTree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      if (side):
        minVal = math.inf
        for move in moves:
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          value, moveList, tree = minimax(newside, newboard, newflags, depth-1)
          moveTree[encode(*move)] = tree
          if value < minVal:
            minVal = value
            minMove = move
            minList = moveList
        minList.insert(0, minMove)
        return (minVal, minList, moveTree)
      else:
        maxVal = -math.inf
        for move in moves:
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          value, moveList, tree = minimax(newside, newboard, newflags, depth-1)
          moveTree[encode(*move)] = tree
          if value > maxVal:
            maxVal = value
            maxMove = move
            maxList = moveList
        maxList.insert(0, maxMove)
        return (maxVal, maxList, moveTree)



def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
      return (evaluate(board), [], {})
    else:
      moveTree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      if (side):
        minVal = math.inf
        for move in moves:
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          value, moveList, tree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
          moveTree[encode(*move)] = tree
          if value < minVal:
            minVal = value
            minMove = move
            minList = moveList
          beta = min(beta, minVal)
          if beta <= alpha:
            break
        minList.insert(0, minMove)
        return (minVal, minList, moveTree)
      else:
        maxVal = -math.inf
        for move in moves:
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          value, moveList, tree = alphabeta(newside, newboard, newflags, depth-1, alpha, beta)
          moveTree[encode(*move)] = tree
          if value > maxVal:
            maxVal = value
            maxMove = move
            maxList = moveList
          alpha = max(alpha, maxVal)
          if alpha >= beta:
            break
        maxList.insert(0, maxMove)
        return (maxVal, maxList, moveTree)
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    if depth == 0:
      return (evaluate(board), [], {})
    else:
      moveTree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      if (side):
        minVal = math.inf
        for move in moves:
          moveTree[encode(*move)] = {}
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          sum = 0
          for i in range(breadth):
            value, list, tree = helper(newside, newboard, newflags, depth-1, chooser)
            sum += value
            moveTree[encode(*move)].update(tree)
          if sum / breadth < minVal:
            minVal = sum / breadth
            minList = list
            minMove = move
        minList.insert(0, minMove)
        return (minVal, minList, moveTree)
      else:
        maxVal = -math.inf
        for move in moves:
          moveTree[encode(*move)] = {}
          newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
          sum = 0
          for i in range(breadth):
            value, list, tree = helper(newside, newboard, newflags, depth-1, chooser)
            sum += value
            moveTree[encode(*move)].update(tree)
          if sum / breadth > maxVal:
            maxVal = sum / breadth
            maxList = list
            maxMove = move
        maxList.insert(0, maxMove)
        return (maxVal, maxList, moveTree)

def helper(side, board, flags, depth, chooser):
  if depth == 0:
    return (evaluate(board), [], {})
  else:
    moveTree = {}
    moves = [ move for move in generateMoves(side, board, flags) ]
    move = chooser(moves)
    newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
    value, moveList, tree = helper(newside, newboard, newflags, depth-1, chooser)
    moveTree[encode(*move)] = tree
    moveList.insert(0, move)
    return (value, moveList, moveTree)