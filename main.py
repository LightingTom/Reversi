import random
import time
import numpy as np

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0
MAX_DEPTH = 4
random.seed(0)

dirct = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, 1), (1, 0), (1, -1)]

corner = [(0, 0, 0), (1, 0, 7), (2, 7, 0), (3, 7, 7)]



# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent
    def __init__(self, chessboard_size, color, time_out):

        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        # the max time you should use, your algorithm's run time must not exceed the time limit. self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your decision .
        self.candidate_list = []
        self.vmap = [[550, -25, 10, 5, 5, 10, -25, 550],
                     [-25, -100, 1, 1, 1, 1, -100, -25],
                     [10, 1, 3, 2, 2, 3, 1, 10],
                     [5, 1, 2, 1, 1, 2, 1, 5],
                     [5, 1, 2, 1, 1, 2, 1, 5],
                     [10, 1, 3, 2, 2, 3, 1, 10],
                     [-25, -100, 1, 1, 1, 1, -100, -25],
                     [550, -25, 10, 5, 5, 10, -25, 550]]
        self.updated = []
        self.s_value = 40
        self.m_value = 10
        self.MAX_Depth = 4

    def valid_pos(self, chessboard, color):
        result = []

        # Here is the simplest sample:Random decision
        idx = np.where(chessboard == COLOR_NONE)
        idx = list(zip(idx[0], idx[1]))
        for i in idx:
            diff_color = 0
            possible = []
            valid = False
            for j in dirct:
                point = (i[0] + j[0], i[1] + j[1])
                if point[0] < 0 or point[1] < 0 or point[0] > 7 or point[1] > 7:
                    continue
                if chessboard[point] == color * -1:
                    diff_color = diff_color + 1
                    possible.append(j)
            if diff_color == 0:
                continue
            else:
                for p in possible:
                    next_point = (i[0] + p[0], i[1] + p[1])
                    while (next_point[0] >= 0 and next_point[1] >= 0 and next_point[0] <= 7 and next_point[1] <= 7 and
                           chessboard[next_point] == color * -1):
                        next_point = (next_point[0] + p[0], next_point[1] + p[1])
                    if next_point[0] >= 0 and next_point[1] >= 0 and next_point[0] <= 7 and next_point[1] <= 7 and \
                            chessboard[next_point] == color:
                        valid = True
            if valid:
                result.append(i)
        return result

    def get_board(self, chessboard, pos, color):
        result = chessboard.copy()
        result[pos] = color
        for i in dirct:
            new_pos = (pos[0] + i[0], pos[1] + i[1])
            while new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] <= 7 and new_pos[1] <= 7 and chessboard[
                new_pos] == color * -1:
                new_pos = (new_pos[0] + i[0], new_pos[1] + i[1])
            if new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] <= 7 and new_pos[1] <= 7 and \
                    chessboard[new_pos] == color:
                d = (-i[0], -i[1])
                new_pos = (new_pos[0] + d[0], new_pos[1] + d[1])
                while new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] <= 7 and new_pos[1] <= 7 and chessboard[
                    new_pos] == color * -1:
                    result[new_pos] = result[new_pos] * -1
                    new_pos = (new_pos[0] + d[0], new_pos[1] + d[1])
        return result

    def stable(self, chessboard, color):
        result = 0
        visit = [0, 0, 0, 0]
        if chessboard[(0, 0)] == color:
            limit = 1
            visit[0] = 1
            result = result + 1
            for i in range(1, 8):
                if chessboard[(i, 0)] == color:
                    result = result + 1
                    limit = limit + 1
                    if i == 7:
                        visit[1] = 1
            for i in range(1, 8):
                if chessboard[(0, i)] == color:
                    result = result + 1
                    for j in range(1, limit):
                        a = chessboard[(j, i)]
                        if chessboard[(j, i)] == color:
                            result = result + 1
                            if (j, i) == (0, 7):
                                visit[2] = 1
                        else:
                            break
                else:
                    break
        if chessboard[(0, 7)] == color and visit[1] == 0:
            result = result + 1
            limit = 1
            for i in range(1, 8):
                if chessboard[(i, 7)] == color:
                    result = result + 1
                    limit = limit + 1
                    if i == 7:
                        visit[3] = 1
            for i in reversed(range(7)):
                if chessboard[(0, i)] == color:
                    result = result + 1
                    for j in range(1, limit):
                        if chessboard[(j, i)] == color:
                            result = result + 1
                        else:
                            break
                else:
                    break
        if chessboard[(7, 0)] == color and visit[2] == 0:
            result = result + 1
            limit = 1
            for i in reversed(range(8)):
                if chessboard[(i, 0)] == color:
                    result = result + 1
                    limit = limit + 1
            for i in range(1, 8):
                if chessboard[(7, i)] == color:
                    result = result + 1
                    if i == 7:
                        visit[3] = 1
                    for j in reversed(range(8 - limit, 8)):
                        if chessboard[(j, i)] == color:
                            result = result + 1
                        else:
                            break
                else:
                    break
        if chessboard[(7, 7)] == color and visit[3] == 0:
            result = result + 1
            limit = 1
            for i in reversed(range(7)):
                if chessboard[(i, 7)] == color:
                    result = result + 1
                    limit = limit + 1
            for i in reversed(range(7)):
                if chessboard[(7, i)] == color:
                    result = result + 1
                    for j in reversed(range(8 - limit, 8)):
                        if chessboard[(j, i)] == color:
                            result = result + 1
                        else:
                            break
                else:
                    break
        return result

    def evaluate(self, chessboard):
        result = 0
        stable_point = self.stable(chessboard, self.color)
        o_sp = self.stable(chessboard,self.color*-1)
        for i in range(8):
            for j in range(8):
                if chessboard[i][j] == self.color:
                    result = result + self.vmap[i][j]
                elif chessboard[i][j] == self.color * -1:
                    result = result - self.vmap[i][j]
        return result + self.s_value * (stable_point - o_sp)+ self.m_value * (
                    len(self.valid_pos(chessboard, self.color)) - len(self.valid_pos(chessboard, self.color * -1)))

    def search(self, chessboard, color, depth):
        # The flag tells that the current state is that
        # we can take the move or the opposite can take the move
        flag = self.color * color
        # If the game ends, we use the number of our chess to be the evaluation score
        if np.sum(chessboard == COLOR_NONE) == 1:
            idx = np.where(chessboard == COLOR_NONE)
            idx = list(zip(idx[0], idx[1]))
            return [(np.sum(chessboard == color),) + idx[0]]
        # If the DFS reach the max depth, do the evaluation
        if depth >= MAX_DEPTH:
            values = []
            for p in self.valid_pos(chessboard, color):
                values.append((self.evaluate(self.get_board(chessboard, p, color)),) + p)
            res = sorted(values, key=lambda x: (x[0] * flag))
            return res
        e_values = []
        v = self.valid_pos(chessboard, color)
        for p in v:
            next_board = self.get_board(chessboard, p, color)
            # Go to the next level
            s_arr = self.search(next_board, color * -1, depth + 1)
            if len(s_arr) != 0:
                best = s_arr[-1]
                e_values.append((best[0],) + p)
        res = sorted(e_values, key=lambda x: (x[0] * flag))
        return res

    def alphabeta(self, chessboard, color, depth, alpha=-float('inf'), beta=float('inf')):
        # If the game ends, we use the number of our chess to be the evaluation score
        if np.sum(chessboard == COLOR_NONE) == 1:
            idx = np.where(chessboard == COLOR_NONE)
            idx = list(zip(idx[0], idx[1]))
            if idx[0] in self.valid_pos(chessboard, color):
                return np.sum(self.get_board(chessboard, idx[0], color) == self.color), idx[0]
            else:
                res = np.sum(chessboard == self.color)
                return res, idx[0]
        # If the DFS reach the max depth, do the evaluation
        if depth > self.MAX_Depth:
            return self.evaluate(chessboard), None
        # Overall max and min value
        max_val = -1000000
        min_val = 1000000
        action = None
        for p in self.valid_pos(chessboard, color):
            # go deeper
            current, p1 = self.alphabeta(self.get_board(chessboard, p, color), color * -1, depth + 1, alpha, beta)
            # in the max-level
            if color == self.color:
                if current > alpha:
                    if current > beta:
                        # after updating, alpha will bigger than beta, so cut
                        return current, p
                    # update alpha if it is the largest so far
                    alpha = current
                # update the overall max value
                if current > max_val:
                    max_val = current
                    action = p
            else:
                # in the min-level
                if current < beta:
                    if current < alpha:
                        # after updating, alpha will bigger than beta, so cut
                        return current, p
                    # update beta to be the minimum so far
                    beta = current
                # update the overall minimum value
                if current < min_val:
                    min_val = current
                    action = p
        if color == self.color:
            return max_val, action
        else:
            return min_val, action

    def refresh(self, number):
        if number == 0:
            self.vmap[0][1] = 15
            self.vmap[1][0] = 15
            self.vmap[1][1] = 15
            self.updated.append(0)
        elif number == 1:
            self.vmap[0][6] = 15
            self.vmap[1][7] = 15
            self.vmap[1][6] = 15
            self.updated.append(1)
        elif number == 2:
            self.vmap[6][0] = 15
            self.vmap[7][1] = 15
            self.vmap[6][1] = 15
            self.updated.append(2)
        elif number == 3:
            self.vmap[6][7] = 15
            self.vmap[7][6] = 15
            self.vmap[6][6] = 15
            self.updated.append(3)

    # The input is current chessboard.
    def go(self, chessboard):
        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        step = np.sum(chessboard != COLOR_NONE) - 3

        if step <= 20:
            self.s_value = 50
        elif step <= 40:
            self.s_value = 30
        else:
            self.s_value = 40

        if 20 <= step <= 40:
            self.m_value = 25
        elif step > 40:
            self.m_value = 15

        if step > 51:
            self.MAX_Depth = 8
        for i in corner:
            if chessboard[(i[1], i[2])] == self.color:
                self.refresh(i[0])
        # ==================================================================
        # Write your algorithm here
        self.candidate_list = self.valid_pos(chessboard, self.color)
        if len(self.candidate_list) >= 8:
            self.MAX_Depth = 3
        else:
            self.MAX_Depth = 4
        result = self.alphabeta(chessboard, self.color, 1)
        if result[1] in self.candidate_list:
            self.candidate_list.append(result[1])
        return self.candidate_list

        # ==============Find new pos========================================
        # Make sure that the position of your decision in chess board is empty.
        # If not, the system will return error.
        # Add your decision into candidate_list, Records the chess board
        # You need add all the positions which is valid
        # candidate_list example: [(3,3),(4,4)]
        # You need append your decision at the end of the candidate_list,
        # we will choose the last element of the candidate_list as the position you choose
        # If there is no valid position, you must return a empty list.

# if __name__ == '__main__':
#     #     ai = AI((8, 8), 1, 1)
#     #     chess = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
#     #                       [0, 0, 0, 0, 0, -1, 0, 0],
#     #                       [0, 0, -1, 0, 0, 0, 0, 0],
#     #                       [0, 0, 0, -1, 1, 1, 0, 0],
#     #                       [0, 0, 0, -1, -1, 0, 0, 0],
#     #                       [0, 0, 0, 0, -1, 0, 0, 0],
#     #                       [0, 0, 0, 0, 0, 0, 0, 0],
#     #                       [0, 0, 0, 0, 0, 0, 0, 0]])
#     #     # chess = ai.get_board(chess,(2,0),1)
#     #     # print(ai.get_board(chess,(2,0),1))
#     #     chess2 = np.array([[-1, -1, -1, 0, 0, 0, 0, 0],
#     #                        [-1, -1, -1, 0, 0, 0, 0, 0],
#     #                        [-1, 1, -1, 0, 0, 1, 0, 0],
#     #                        [0, 0, -1, 1, 1, 0, 0, 0],
#     #                        [0, 1, 1, 1, 1, 0, 0, 0],
#     #                        [0, 0, -1, -1, 1, 1, 0, 0],
#     #                        [0, 0, 0, 0, 0, -1, 1, 0],
#     #                        [0, 0, 0, 0, 0, -1, 0, 0]])
#     #     # print(ai.valid_pos(chess, 1))
#     #     # time1 = time.time()
#     #     print(ai.search(chess, 1, 1))
#         time2 = time.time()
#         ai2 = AI((8, 8), 1, 1)
#         chess3 = np.array([[0, 0, -1, -1, -1, 0, 0, 0],
#                       [0, 0, -1, -1, 0, 0, 0, 0],
#                       [0, 0, -1, -1, 1, 0, 1, 0],
#                       [0, 0, 1, -1, 1, -1, -1, 0],
#                       [0, 0, 0, 1, 1, -1, -1, 0],
#                       [0, 0, 1, 1, 1, -1, -1, 0],
#                       [0, 0, 1, 1, 0, -1, -1, 0],
#                       [0, 0, 1, 1, 1, 0, -1, 1]])
#         print(ai2.alphabeta(chess3, 1,1))
#         time3 = time.time()
# #     # print(time2 - time1)
#         print(time3 - time2)
