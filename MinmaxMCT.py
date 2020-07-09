import sys, time, math
import copy
import random

#------------------------------------ MCT vs Minimax ------------------------------------
#Decribe Variable
#These variables for players
HINT_TILE = 'HINT_TILE'
MCTS_NUM = 2
debug = False
player = 1
victory = 0
whiteTiles = 2
blackTiles = 2
useAI = True
changed = True
AIMinmax = True
AIMCT = False
move = (-1, -1)
MAX_THINK_TIME = 20
mTime= 0
mcTime=0

#These variables for board
board = [[0 for x in range(8)] for x in range(8)]
board[3][3] = 1
board[3][4] = 2
board[4][3] = 2
board[4][4] = 1

#newGame is starting method
def newGame():
    # I said global for variable to access it.
    global changed
    global mTime
    global mcTime
    # This loop over until the victory change that means the game is over.
    while victory ==0 :
        #If AIMinMax is True go to MinmaxMove method
        if AIMinmax:
            startMin = time.time()
            AIMinMaxMove()
            endMin = time.time()
            print('Evaluation time for Minmax: {}s'.format(round(endMin - startMin, 7)))
            mTime +=endMin - startMin
        # If AIMCT is True go to MCTMove
        elif AIMCT:
            start = time.time()
            AIMCTMove()
            end = time.time()
            print('Evaluation time for MCT: {}s'.format(round(end - start, 7)))
            mcTime +=end - start
        if changed:
            changed = False

        if AIMinmax:
            startMin = time.time()
            AIMinMaxMove()
            endMin = time.time()
            print('Evaluation time for Minmax: {}s'.format(round(endMin - startMin, 7)))
            mTime += endMin - startMin

        if AIMCT:
            start = time.time()
            AIMCTMove()
            end = time.time()
            print('Evaluation time for MCT: {}s'.format(round(end - start, 7)))
            mcTime += end - start

def performMove(x, y):
    global changed
    global player
    global debug
    global victory
    global whiteTiles
    global blackTiles

    numFlipped = isAvaible(board, x, y, player, PLAYMODE=True)
    if debug:
        print("Flipped " + str(numFlipped) + " pieces!")
    changed = True

    #check the game is over
    allTiles = [item for sublist in board for item in sublist]
    emptyTiles = sum(1 for tile in allTiles if tile == 0)
    whiteTiles = sum(1 for tile in allTiles if tile == 2)
    blackTiles = sum(1 for tile in allTiles if tile == 1)
    print("Current state - empty: " + str(emptyTiles) + " white: " + str(
        whiteTiles) + " black: " + str(blackTiles))

    if debug:
        for x in range(0, 8):
            for y in range(0, 8):
                print(str(board[x][y]) + " ", end='')
            print('')

    if whiteTiles < 1 or blackTiles < 1 or emptyTiles < 1:
        if whiteTiles > blackTiles:
            victory = 2
        elif whiteTiles < blackTiles:
            victory = 1
        else:
            victory = -1
        changed = True
        whiteTiles = whiteTiles
        blackTiles = blackTiles
        return
    movesFound = moveCanBeMade(board, 3 - player)
    if not movesFound:
        if debug:
            print("Player " + str(3 - player) + " cannot move!")
        movesFound = moveCanBeMade(board, player)
        if not movesFound:
            if debug:
                print("Player " + str(player) + "cannot move either!")
            if whiteTiles > blackTiles:
                victory = 2
            elif whiteTiles < blackTiles:
                victory = 1
            else:
                victory = -1
            changed = True
            whiteTiles = whiteTiles
            blackTiles = blackTiles
            return
        else:
            if debug:
                print("Player " + str(player) + " can move, then move!")
            if useAI and player == 2:
                performMoveMinMax()
            changed = True
    else:
        player = 3 - player
        changed = True

#It is check the move can be made with isAvaible method for Minimax.
def moveCanBeMade(board, playerID):
    movesFound = False
    for row in range(0, 8):
        for col in range(0, 8):
            if movesFound:
                continue
            elif board[row][col] == 0:
                numAvailableMoves = isAvaible(board, row, col, playerID, PLAYMODE=False)
                if numAvailableMoves > 0:
                    movesFound = True
    return movesFound

#This Method for Minimax Move
def AIMinMaxMove():
    global AIMinmax
    global AIMCT

    performMoveMinMax()
    AIMinmax = False
    AIMCT = True

#This Method for MCT Move
def AIMCTMove():
    current_path = []  # tofind current path
    root = expanding()  # I create an expand node for player 2 that is AI
    root_copy = expanding()
    mcts_search(current_path, root_copy, root)
    maxValue = -1
    result = (0, 0)
    #MCT move coordinates
    for n in root:
        mct_move, lay, win, child = n
        if (lay > 0) and (win / lay > maxValue):
            result = mct_move
            maxValue = win / lay

    x = result[0]
    y = result[1]
    performMove(x, y)
    global AIMCT
    global AIMinmax

    AIMCT = False
    AIMinmax = True

# Selecting- Search the Tree
def mcts_search(current_path, root_copy, root):
    isMCTS = True
    for loop in range(0, 500):  # It is iteration.
        # -------- Find Path -----------
        while True:
            if len(root_copy) == 0:
                break
            else:
                list = [0]
                index = 0
                if isMCTS:
                    bestNode = -1
                else:
                    bestNode = 2
                for n in root_copy:
                    if isMCTS:
                        list, bestNode = findMaxBestNode(n, loop, bestNode, list, index)
                    else:
                        list, bestNode = findMinBestNode(n, loop, bestNode, list, index)
                    index += 1
                mct_move, lay, win, child = root_copy[simulating(list)]
                current_path.append(mct_move)
                loop = lay
                root_copy = child
                isMCTS = not (isMCTS)

        current_children = root
        for i in current_path:
            count = 0
            for n in current_children:
                mct_move, lay, win, child = n
                if i[0] == mct_move[0] and i[1] == mct_move[1]:
                    break
                count += 1

            if i[0] == mct_move[0] and i[1] == mct_move[1]:
                lay += 1
                if lay >= 5 and len(child) == 0:
                    child = expanding()
                current_children[count] = (mct_move, lay, win, child)
            current_children = child

def findMaxBestNode(n_tuple, loop, maxval, maxidxlist, index):
    # Calculate UCB
    cval = ucb(n_tuple, loop, 0.1)
    # Add calculateion result in a list
    if cval >= maxval:
        if cval == maxval:
            maxidxlist.append(index)
        else:
            maxidxlist = [index]
            maxval = cval
    return maxidxlist,maxval

def findMinBestNode(n_tuple, loop, maxval, maxidxlist, index):
    # Calculate UCB
    cval = ucb(n_tuple, loop, -0.1)
    # Add calculateion result in a list
    if cval <= maxval:
        if cval == maxval:
            maxidxlist.append(index)
        else:
            maxidxlist = [index]
            maxval = cval
        return maxidxlist,maxval

#Simulating
def simulating(list):
    randValue = list[random.randrange(0, len(list))] # select randomly
    return randValue

#This method checks the game rules.
def isAvaible(board, row, col, playerID, PLAYMODE=True):
    global changed
    global player
    global debug
    global victory
    global whiteTiles
    global blackTiles

    if PLAYMODE:
        board[row][col] = player
    count = 0
    __column = board[row]
    __row = [board[i][col] for i in range(0, 8)]
    if playerID in __column[:col]:
        changes = []
        searchCompleted = False
        for i in range(col - 1, -1, -1):
            if searchCompleted:
                continue
            piece = __column[i]
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append(i)
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i in changes:
                    board[row][i] = player


    if playerID in __column[col:]:
        changes = []
        searchCompleted = False

        for i in range(col + 1, 8, 1):
            if searchCompleted:
                continue
            piece = __column[i]
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append(i)
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i in changes:
                    board[row][i] = player



    if playerID in __row[:row]:
        changes = []
        searchCompleted = False

        for i in range(row - 1, -1, -1):
            if searchCompleted:
                continue
            piece = __row[i]
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append(i)
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i in changes:
                    board[i][col] = player



    if playerID in __row[row:]:
        changes = []
        searchCompleted = False

        for i in range(row + 1, 8, 1):
            if searchCompleted:
                continue
            piece = __row[i]
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append(i)
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i in changes:
                    board[i][col] = player

    i = 1
    ulDiagonal = []
    while row - i >= 0 and col - i >= 0:
        ulDiagonal.append(board[row - i][col - i])
        i += 1
    if playerID in ulDiagonal:
        changes = []
        searchCompleted = False
        for i in range(0, len(ulDiagonal)):
            piece = ulDiagonal[i]
            if searchCompleted:
                continue
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append((row - (i + 1), col - (i + 1)))
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i, j in changes:
                    board[i][j] = player



    i = 1
    urDiagonal = []
    while row + i < 8 and col - i >= 0:
        urDiagonal.append(board[row + i][col - i])
        i += 1
    if playerID in urDiagonal:
        changes = []
        searchCompleted = False
        for i in range(0, len(urDiagonal)):
            piece = urDiagonal[i]
            if searchCompleted:
                continue
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append((row + (i + 1), col - (i + 1)))
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i, j in changes:
                    board[i][j] = player


    i = 1
    llDiagonal = []
    while row - i >= 0 and col + i < 8:
        llDiagonal.append(board[row - i][col + i])
        i += 1
    if playerID in llDiagonal:
        changes = []
        searchCompleted = False

        for i in range(0, len(llDiagonal)):
            piece = llDiagonal[i]
            if searchCompleted:
                continue
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append((row - (i + 1), col + (i + 1)))
        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i, j in changes:
                    board[i][j] = player


    i = 1
    lrDiagonal = []
    while row + i < 8 and col + i < 8:
        lrDiagonal.append(board[row + i][col + i])
        i += 1
    if playerID in lrDiagonal:
        changes = []
        searchCompleted = False

        for i in range(0, len(lrDiagonal)):
            piece = lrDiagonal[i]
            if searchCompleted:
                continue
            if piece == 0:
                changes = []
                searchCompleted = True
            elif piece == playerID:
                searchCompleted = True
            else:
                changes.append((row + (i + 1), col + (i + 1)))

        if searchCompleted:
            count += len(changes)
            if PLAYMODE:
                for i, j in changes:
                    board[i][j] = player

    if count == 0 and PLAYMODE:
        board[row][col] = 0

    return count

#Calculate UCB
def ucb(node, t, cval):
    name, lay, win, child = node
    if lay == 0:
        lay = 0.00000000001 #when you divide zero it will be infinity
    if t == 0:
        t = 1 #log0 is can not calculate
    return (win / lay) + cval * math.sqrt(2 * math.log(t) / lay)

#Expanding
def expanding():
    place_count = []
    result = []
    # This for loop controls avaible positions
    for i in range(0, 8):
        for j in range(0, 8):
            if board[i][j] != 0:
                continue
            if isAvaible(board, i, j, 2, PLAYMODE=False) > 0:
                place_count.append((i, j))
    for n in place_count:
        result.append((n, 0, 0, []))
    return result

#This method for Minimax Algorithm.
def performMoveMinMax():
    tmpBoard = [row[:] for row in board]
    startTime = time.time()
    timeElapsed = 0
    depth = 2
    optimalMove = (-1, -1)
    optimalBoard = tmpBoard
    stop = False #this value for ending the minmax algorithm. If the successboard is true then stop the minmax algorithm.
    currentLevel =0
    # I did 'timeElapsed < 5' because the minimax algorithm time will be less than 5 in a move.
    while not stop and timeElapsed < 5:
        (stop, optimalBoard) = miniMax(tmpBoard, currentLevel, depth, player , -math.inf, math.inf,stop)
        endTime = time.time()
        timeElapsed += endTime - startTime
        startTime = endTime
        #depth += 1 (actually when you increase depth in every move, the minimax algorithm gives better results.)

    for row in range(0, 8):
        for col in range(0, 8):
            if tmpBoard[row][col] != optimalBoard[row][col]:
                optimalMove = (row, col)

    move = optimalMove
    performMove(move[0], move[1])

#For minimax algorithm
def miniMax(board, currentLevel, maxLevel, player, alpha, beta,stop):
    all = [item for sublist in board for item in sublist]
    white = sum(1 for tile in all if tile == 2)
    black = sum(1 for tile in all if tile == 1)
    successBoards = []

    if (not moveCanBeMade(board, player) or currentLevel == maxLevel):
        return (stop, board)
    if white > black:
        diff = (white / (black + white)) * 100
    else:
        diff = - (black / (black + white)) * 100
    # Mobility controls how many steps can player move.
    # first player is black
    # second player is white
    if moveCanBeMade(board, 1) + moveCanBeMade(board, 2) == 0:
        mobility = 0
    else:
        mobility = 100 * moveCanBeMade(board, 2) / (moveCanBeMade(board, 2) + moveCanBeMade(board, 1))

    # Update the sucessorBoard
    for row in range(0, 8):
        for col in range(0, 8):
            if board[row][col] == 0:
                numAvailableMoves = isAvaible(board, row, col, player, PLAYMODE=False)
                if numAvailableMoves > 0:
                    successBoard = [row[:] for row in board]
                    successBoard[row][col] = player
                    successBoards.append(successBoard)

    if len(successBoards) == 0:
        stop = True
        return (stop, board)
    bestBoard = None

    # If player is 1 that means white, check the alpha value.
    # If the value is smaller than best update bestBoard
    if player == 1:
        maxValue = -math.inf #alpha
        for i in range(0, len(successBoards)):
            stop, boardS = miniMax(successBoards[i], currentLevel + 1, maxLevel, 2, alpha,beta,stop)
            best = diff + mobility  #best is heuristics for nonfinal node.
            if best > maxValue:
                maxValue = best
                bestBoard = successBoards[i]
            alpha = max(alpha, best)
            if best >= beta:
                return (stop, boardS)
    # If player is 2 that means white, check the beta value.
    # If the value is greather than utility update bestBoard
    else:
        minValue = math.inf
        for i in range(0, len(successBoards)):
            stop, boardS = miniMax(successBoards[i], currentLevel + 1, maxLevel, 1, alpha,beta,stop)
            best = diff + mobility
            if best < minValue:
                minValue = best
                bestBoard = successBoards[i]
            beta = min(beta, best)
            if best <= alpha:
                return (stop, boardS)

    return (stop, bestBoard)

if __name__ == '__main__':
    newGame()
    if victory == 2:
        print("MCT Won!")
    elif victory == 1:
        print("Min-Max Won!")
    elif victory == -1:
        print("Draw!")

