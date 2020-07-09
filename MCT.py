import pygame
import sys, time, math
from collections import deque
import random
import copy

#------------------------------------ MCT vs Human ------------------------------------
#Decribe Variable
#These variables for board game gui
BLOCK_SIZE = 50
PADDING_SIZE = 5
BOARD_WIDTH = BLOCK_SIZE * 8
BOARD_HEIGHT = BLOCK_SIZE * 8 + 20
FRAME_PER_SECOND = 40
WINDOWWIDTH = BLOCK_SIZE * 10
WINDOWHEIGHT = BLOCK_SIZE * 10
XMARGIN = int(((8 * BLOCK_SIZE)) / 2)
YMARGIN = int(((8 * BLOCK_SIZE)) / 2)
HINT_TILE = 'HINT_TILE'
retract_moves = deque(maxlen=5)

#These variables for players
MCTS_NUM = 2
MAX_THINK_TIME = 5
debug = False
player = 1
victory = 0
whiteTiles = 2
blackTiles = 2
useAI = True
changed = True
AIReadyToMove = False

#These variables for board
board = [[0 for x in range(8)] for x in range(8)]
board[3][3] = 1
board[3][4] = 2
board[4][3] = 2
board[4][4] = 1

#Creating a board
pygame.init()
screen = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption('Othello')
font = pygame.font.SysFont("Helvetica", 48)
icon = pygame.image.load("boardIcon.png")
pygame.display.set_icon(icon)
boardground = pygame.image.load('boardground.png')
black = (pygame.image.load('black.png'))
white = pygame.image.load('white.png')
BGIMAGE = pygame.image.load('background.jpg')
BGIMAGE = pygame.transform.smoothscale(BGIMAGE, (WINDOWWIDTH, WINDOWHEIGHT))
black_color= (0,0,0)
menuFont = pygame.font.SysFont("comicsansms", 15)

#newGame is starting method
def newGame():
    global changed #I said global for variable to access it.
    # This loop over until the player exit.
    while True:
        # If AI move is True, go to the AIMove method.
        if AIReadyToMove:
            start = time.time()
            AIMove()
            end = time.time()
            print('Evaluation time: {}s'.format(round(end - start, 7)))
        # Otherwise it is human so I have to control its mouse click.
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    x, y = event.pos
                    if x >= 120 and x <= 175 and y >= 480:
                        newGame()
                    elif x >= 220 and x <= 250 + menuFont.size("Exit")[0] and y > 480:
                        pygame.quit()
                        sys.exit()
                    elif x >= 50 and x <= 450 and y >= 50 and y <= 450:
                        # I control the board x and y coordinates. When you run this game, there will be a board.
                        # The board is actual board and there will be also a board that is game board inside the actual borad
                        chessman_x = int(math.floor(x / (BLOCK_SIZE)) - 1) #convert RealBoard size to GameBoard
                        chessman_y = int(math.floor(y / (BLOCK_SIZE)) - 1) #convert RealBoard size to GameBoard
                        if debug:
                            print("player " + str(player) + " x: " + str(chessman_x) + " y: " + str(
                                chessman_y))
                        try:
                            playerMove(chessman_x, chessman_y)
                        except Exception as e:
                            print("Wrong Place")
                else:
                    pass
        if changed:
            drawBoard()
            changed = False

        if AIReadyToMove:
             AIMove()
             drawBoard()

        clock.tick(FRAME_PER_SECOND)
    self.quitGame()

#This method for text on the actual board.
def drawText(text, font, screen, x, y, rgb):
    textObj = font.render(text, 1, (rgb[0],rgb[1],rgb[2]))
    textRect = textObj.get_rect()
    textRect.topleft = (x, y)
    screen.blit(textObj, textRect)

#This method draw the board in every change
def drawBoard():
    # These are for board background image and menu.
    background = pygame.Rect(0,0,WINDOWWIDTH,WINDOWHEIGHT)
    screen.blit(BGIMAGE, background)
    boardTaabla = pygame.Rect(BLOCK_SIZE, BLOCK_SIZE, BOARD_WIDTH, BOARD_HEIGHT)
    screen.blit(boardground, boardTaabla)
    menu = pygame.Rect(0, 480, WINDOWWIDTH, 20)
    pygame.draw.rect(screen, (255, 255, 255), menu)
    menuFont = pygame.font.SysFont("comicsansms", 15)
    drawText("Restart", menuFont, screen, 120, 480 - 1, (0, 0, 0))
    drawText("Exit", menuFont, screen,220, 480 - 1, (0, 0, 0))
    drawText("Hint", menuFont, screen, 320, 480 - 1, (0, 0, 0)) ## However it is not working

    # These line for GameBoard
    for i in range(9):
        startx = (i * 50) + BLOCK_SIZE
        starty = BLOCK_SIZE
        endx = (i * 50) + BLOCK_SIZE
        endy = BLOCK_SIZE + (8 * 50)
        pygame.draw.line(screen, black_color, (startx, starty), (endx, endy))
    for i in range(9):
        startx = BLOCK_SIZE
        starty = (i * 50) + BLOCK_SIZE
        endx = BLOCK_SIZE + (8 * 50)
        endy = (i * 50) + BLOCK_SIZE
        pygame.draw.line(screen, black_color, (startx, starty), (endx, endy))

    # gameBoard(boad) is 2D array.
    # If an element in the array is 1 that means the element is black otherwise white.
    for row in range(0, 8):
        for col in range(0, 8):
            block = board[row][col]
            chessman_size = BLOCK_SIZE - 2 * PADDING_SIZE
            chessman = pygame.Rect(row * BLOCK_SIZE + PADDING_SIZE + BLOCK_SIZE, col * BLOCK_SIZE + PADDING_SIZE + BLOCK_SIZE,chessman_size, chessman_size)
            if block == 1:
                screen.blit(black, chessman)
            elif block == 2:
                screen.blit(white, chessman)
            elif block == 0:
                pass
            else:
                sys.exit('Error occurs - player number incorrect!')

    if victory == -1:
        drawText("Draw! " + str(whiteTiles) + ":" + str(blackTiles), font, screen, 50, 10, (255, 128, 0))
    elif victory == 1:
        if useAI:
            drawText("You Won! " + str(blackTiles) + ":" + str(whiteTiles),font, screen, 50, 10, (255, 128, 0))
    elif victory == 2:
        if useAI:
            drawText("MCT Won! " + str(whiteTiles) + ":" + str(blackTiles), font, screen, 50, 10, (255, 128, 0))

    pygame.display.update()

#I check the player's move
def playerMove(x, y):
    global AIReadyToMove
    if victory != 0 or (useAI and player != 1):
        return
    performMove(x, y)
    if useAI and player == 2:
        AIReadyToMove = True
        if debug:
            print("AI is ready to move!")

def performMove( x, y):
    global changed
    global victory
    global blackTiles
    global whiteTiles
    global player
    global AIReadyToMove

    if board[x][y] != 0:
        print(" - Block has already been occupied!")
    else:
        numFlipped = isAvaible(board, x, y, player, PLAYMODE=True)
        if debug:
            print("Flipped " + str(numFlipped) + " pieces!")
        changed = True

        # check game ending
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
                    AIMove()
                    AIReadyToMove = False
                    changed = True
        else:
            player = 3 - player
            changed = True

#It is check the move can be made with isAvaible method.
def moveCanBeMade(board, playerID):
    movesFound = False
    for row in range(0, 8):
        for col in range(0, 8):
            if movesFound:
                continue
            elif board[row][col] == 0:
                if isAvaible(board, row, col, playerID, PLAYMODE=False) > 0:
                    movesFound = True
    return movesFound

#This method goes to AI.
def AIMove():
    current_path = []  # tofind current path
    root = expanding()  # I create an expand node for player 2 that is AI
    root_copy = expanding()
    mcts_search(current_path, root_copy, root)
    maxValue = -1
    result = (0, 0)
    # MCT move coordinates
    for n in root:
        mct_move, lay, win, child = n
        if (lay > 0) and (win / lay > maxValue):
            result = mct_move
            maxValue = win / lay

    x = result[0]
    y = result[1]
    performMove(x,y)
    global AIReadyToMove
    AIReadyToMove = False

################ MCT ALGORITHM ########################

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

def findMaxBestNode(node, loop, max, list, index):
    # Calculate UCB
    cval = ucb(node, loop, 0.1)
    # Add calculateion result in a list
    if cval >= max:
        if cval == max:
            list.append(index)
        else:
            list = [index]
            max = cval
    return list,max

def findMinBestNode(node, loop, min, list, index):
    # Calculate UCB
    cval = ucb(node, loop, -0.1)
    # Add calculateion result in a list
    if cval <= min:
        if cval == min:
            list.append(index)
        else:
            list = [index]
            min = cval
        return list,min

#Simulating
def simulating(list):
    randValue = list[random.randrange(0, len(list))] # select randomly
    return randValue

def isAvaible(board, row, col, playerID, PLAYMODE=True):
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
def ucb(node, t, value):
    name, lay, win, child = node
    if lay == 0:
        lay = 0.00000000001  # when you divide zero it will be infinity
    if t == 0:
        t = 1  # log0 is can not calculate
    return (win / lay) + value * math.sqrt(2 * math.log(t) / lay)

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

if __name__ == '__main__':
    newGame()