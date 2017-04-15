#!/bin/env python3
import os
import sys
import time
import cv2
import numpy as np

from copy import deepcopy as copy
from itertools import groupby
from enum import Enum
from threading import Thread
from pykeyboard import PyKeyboardEvent
from pymouse import PyMouse
from time import sleep

mouse = PyMouse()

class State(Enum):
    FIND_RECT = 1
    WAIT_START = 2
    WAIT_MOTION = 3
    WAIT_POINT = 4
    MAKE_MOVE = 5

class MoveState(Enum):
    NEW = 0,
    WAIT = 1,
    DONE = 2

def score(arr):
    res = 0
    for line in arr:
        res = max(max([sum(1 for _ in g) for _,g in groupby(line)]), res)

    arr = np.rot90(arr)
    for line in arr:
        res = max(max([sum(1 for _ in g) for _,g in groupby(line)]), res)

    return res

def make_move(b, game_x, game_y, box_w, box_h):
    global move_state

    off_x = int(sys.argv[1])
    off_y = int(sys.argv[2])

    if move_state != MoveState.WAIT:
        print("MoveState isn't WAIT!")
        return

    if score(b) > 2:
        move_state = MoveState.DONE
        return

    mouse.click(off_x + int(game_x*1.25) + 20,
            off_x + int(game_y*1.25) + 20)
    m_sc = 2
    m_x1 = 0
    m_y1 = 0
    m_x2 = 0
    m_y2 = 0

    for y in range(len(b)):
        for x in range(len(b[0])):
            if x+1 < len(b[0]):
                m = copy(b)
                m[y][x], m[y][x+1] = m[y][x+1], m[y][x]
                sc = score(m)
                if sc > m_sc:
                    m_sc = sc
                    m_x1, m_y1 = x+1, y
                    m_x2, m_y2 = x, y
            if x-1 > 0:
                m = copy(b)
                m[y][x], m[y][x-1] = m[y][x-1], m[y][x]
                sc = score(m)
                if sc > m_sc:
                    m_sc = sc
                    m_x1, m_y1 = x-1, y
                    m_x2, m_y2 = x, y
            if y+1 < len(b):
                m = copy(b)
                m[y][x], m[y+1][x] = m[y+1][x], m[y][x]
                sc = score(m)
                if sc > m_sc:
                    m_sc = sc
                    m_x1, m_y1 = x, y+1
                    m_x2, m_y2 = x, y
            if y-1 > 0:
                m = copy(b)
                m[y][x], m[y-1][x] = m[y-1][x], m[y][x]
                sc = score(m)
                if sc > m_sc:
                    m_sc = sc
                    m_x1, m_y1 = x, y-1
                    m_x2, m_y2 = x, y

    print("Best move: Score: %d from (%d,%d) to (%d,%d)"
            % (m_sc, m_x1, m_y1, m_x2, m_y2))


    mouse.click(off_x + int((game_x + int(box_w*(m_x1+0.5)))*1.25),
                off_y + int((game_y + int(box_w*(m_y1+0.5)))*1.25))
    sleep(0.06)
    mouse.click(off_x + int((game_x + int(box_w*(m_x2+0.5)))*1.25),
                off_y + int((game_y + int(box_w*(m_y2+0.5)))*1.25))
    sleep(0.06)

    move_state = MoveState.DONE

def begin():
    global move_state
    move_state = MoveState.NEW

    w = int(sys.argv[3])
    h = int(sys.argv[4])

    state = State.FIND_RECT
    game_rect = None
    timer = None

    # Used for motion detection
    last = None

    print('-> record')
    fileno = sys.stdin.fileno()
    while 1:
        data = os.read(fileno, w * h * 3)
        image =  np.fromstring(data, dtype='uint8')

        # Skip invalid frames
        if len(image) < w*h*3:
            continue

        image = image.reshape((h, w, 3))
        image = cv2.resize(image, (int(w*0.8), int(h*0.8)))
        image = cv2.GaussianBlur(image, (5, 5), 0)
        #_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        black = np.zeros((int(h*0.8), int(w*0.8), 3), np.uint8)

        # Do operations here
        # Remember all colors in cv2 are BGR

        # Start by finding game rectangle
        if state == State.FIND_RECT:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            (_, cnts, _) = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if our approximated contour has four points, then
                # we can assume that we have found our game rect
                if len(approx) == 4 and (approx[1,0,1] - approx[0,0,1]) > h//3:
                    game_rect = approx
                    state = State.WAIT_START
                    break

        # Wait for the game to start
        if state == State.WAIT_START:
            if timer == None:
                timer = time.time()
            elif (time.time() - timer) > 2.1: # Should wait two seconds
                timer = None
                state = State.WAIT_MOTION

        # Wait for stuff to stop moving (check 20 frames)
        if state == State.WAIT_MOTION:
            moved = False
            new = image[game_rect[0,0,1]:game_rect[1,0,1], game_rect[0,0,0]:game_rect[2,0,0]]
            new = cv2.resize(new, (100, 100)) # Downscaling for perf
            new = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
            if last is None:
                last = new
            else:
                res = cv2.countNonZero(cv2.absdiff(last, new))
                if res == 0:
                    if moved:
                        state = State.WAIT_POINT
                    else:
                        state = State.MAKE_MOVE

                last = new
                moved = True

        # Wait to see if more stuff falls
        if state == State.WAIT_POINT:
            if timer == None:
                timer = time.time()
            elif (time.time() - timer) > 0.05:
                timer = None
                state = State.WAIT_MOTION

        if state == State.MAKE_MOVE:
            mtop = 30 # margin
            mbot = 5
            mlft = 7
            mrgt = 7

            game_x = game_rect[0,0,0]
            game_y = game_rect[0,0,1]
            box_w = (game_rect[2,0,0] - game_rect[0,0,0])//8
            box_h = (game_rect[1,0,1] - game_rect[0,0,1])//8

            board = []

            for x in range(8):
                for y in range(8):
                    blx1, bly1 = (game_x + box_w*x + mlft), (game_y + box_h*y + mtop)
                    blx2, bly2 = (game_x + box_w*(x+1) - mrgt), (game_y + box_h*(y+1) - mbot)

                    bpix = image[bly1:bly2, blx1:blx2]
                    bpix = cv2.GaussianBlur(bpix, (5, 5), 0)
                    #_, bpix = cv2.threshold(bpix, 127, 255, cv2.THRESH_BINARY)

                    color = np.uint8(np.average(np.average(bpix, axis=0), axis=0))

                    # Decompose colors into True / False
                    b = (color[0] > 127)
                    g = (color[1] > 127)
                    r = (color[2] > 127)

                    board.append((r << 2) | (g << 1) | b)

                    # Workaround for lion
                    if r and not g and not b:
                        if  color[0] < 8:
                            board[-1] = 9

                    cv2.rectangle(black, (blx1, bly1), (blx2, bly2),
                            (int(b)*255,
                             int(g)*255,
                             int(r)*255))

            # Break results in chunks and rotate it to get baord
            board = [board[i:i + 8] for i in range(0, len(board), 8)]
            board = [inv[::-1] for inv in board]
            board = np.rot90(board)

            if move_state == MoveState.NEW:
                move_state = MoveState.WAIT
                Thread(target=make_move, args=(board, game_x, game_y,
                    box_w, box_h)).start()
            elif move_state == MoveState.DONE:
                move_state = MoveState.NEW
                state = State.MAKE_MOVE

        # Draw debug info
        cv2.drawContours(black, game_rect, -1, (255, 255, 0), 3)
        cv2.putText(black, str(state)[6:], (6,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))

        cv2.imshow('black', black)
        cv2.imshow('hsklive', image)

        # esc to quit
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    begin()
