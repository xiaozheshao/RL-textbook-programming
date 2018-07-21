import sys
import numpy as np
import matplotlib.pyplot as plt


GAMMA = 1
WIDTH = 17
HEIGHT = 24


class Environment(object):
    def __init__(self):
        self.trace = np.reshape(np.fromfile("trace.txt", sep = ' '), (-1, WIDTH))
        self.row, self.column = np.shape(self.trace)
        print(self.trace)
        print(self.row, self.column)
        self.rowspeedlimit = 5
        self.columnspeedlimit = 5
        self.rowspeed = 0
        self.columnspeed = 0
        self.rowindex = 0
        self.columnindex = 0


    def rspeedlimit(self):
        return self.rowspeedlimit

    def cspeedlimit(self):
        return self.columnspeedlimit

    def rowsize(self):
        return self.row

    def columnsize(self):
        return self.column

    def actionsize(self):
        return 9

    def ishit(self, x, y):
        deltax = self.rowindex - x
        deltay = y - self.columnindex
        if deltax >= deltay:
            flag = True
            if deltax > 0:
                inc = float(deltay) / deltax
                increase = (1, inc)
                length = deltax
            else:
                inc = 0
                increase = (0, inc)
                length = 0
        else:
            flag = False
            inc = float(deltax) / deltay
            increase = (inc, 1)
            length = deltay

#        print("increase", increase)
        currentx = float(self.rowindex)
        currenty = float(self.columnindex)
#        print("currentx", currentx)
#        print("currenty", currenty)
        if currentx < 0:
            # out of range
            return True
        for l in range(0, length + 1):
            if self.trace[int(currentx)][int(currenty)] == 0:
                return True
            elif self.trace[int(currentx)][int(currenty)] == 2:
                return False
            currentx -= increase[0]
            currenty += increase[1]



    def takeaction(self, action):
        racc = int(action / 3) - 1
        cacc = int(action % 3) - 1
#        print("racc", racc)
#        print("cacc", cacc)
        self.rowspeed += racc
        if self.rowspeed < 0:
            self.rowspeed = 0
        elif self.rowspeed > self.rowspeedlimit - 1:
            self.rowspeed = self.rowspeedlimit - 1
        self.columnspeed += cacc
        if self.columnspeed < 0:
            self.columnspeed = 0
        elif self.columnspeed > self.columnspeedlimit - 1:
            self.columnspeed = self.columnspeedlimit - 1
        x = self.rowindex - self.rowspeed
        y = self.columnindex + self.columnspeed
#        print("row speed", self.rowspeed)
#        print("column speed", self.columnspeed)
#        print("x", x)
#        print("y", y)
#        assert(HEIGHT > x >= 0)
#        assert(WIDTH > y >= 0)
        if self.ishit(x, y):
            s = self.reset()
        else:
            self.rowindex = x
            self.columnindex = y
            s = (x,y, self.rowspeed, self.columnspeed)
        #if self.trace[x][y] == 2:
        if x <= 5 and y >= WIDTH:
            done = True
            reward = 0
        else:
            done = False
            reward = -1
        return reward, s, done

    def reset(self):
        num = np.count_nonzero(self.trace[-1])
        first = np.nonzero(self.trace[-1])
#        print("num", num)
#        print("first[0][0]", first[0][0])
        c = np.random.randint(first[0][0], first[0][0] + num)
        assert(self.trace[self.rowsize() - 1, c] == 1)
        self.rowspeed = 0
        self.columnspeed = 0
        self.rowindex = self.rowsize() - 1
        self.columnindex = c
        return (self.rowsize() - 1, c, 0, 0)

class Agent(object):
    def __init__(self, row, column, rspeedlimit, cspeedlimit, actionsize, env, epsilon=0.1):
        print(row, column, actionsize)
        self.Q = np.random.rand(row, column, rspeedlimit, cspeedlimit, actionsize) - 10000
        self.C = np.zeros((row, column, rspeedlimit, cspeedlimit, actionsize))
        self.PI = np.zeros((row, column, rspeedlimit, cspeedlimit), dtype=int)
        self.epsilon = epsilon
        self.env = env
        self.actionsize = actionsize
        self.row = row
        self.column = column
        self.rspeedlimit = rspeedlimit
        self.cspeedlimit = cspeedlimit
        self.updatePI()
#        print(self.PI)

    def updatePI(self):
        for n in range(0, self.row):
            for m in range(0, self.column):
                for ns in range(0, self.rspeedlimit):
                    for ms in range(0, self.cspeedlimit):
                        self.PI[n][m][ns][ms] = np.argmax(self.Q[n][m][ns][ms])

    def genepisode(self):
        # generate an episode based on an epsilon-soft policy
        self.S = []
        self.A = []
        self.R = [0]
        self.b = []
        state = self.env.reset()
        self.S.append(state)
        while True:
            randnum = np.random.rand(1)
            if randnum < self.epsilon:
                action = np.random.randint(0, self.actionsize)
                if action == self.PI[state[0]][state[1]][state[2]][state[3]]:
                    self.b.append((1 - self.epsilon) + self.epsilon / self.actionsize)
                else:
                    self.b.append(self.epsilon / self.actionsize)
            else:
                action = self.PI[state[0]][state[1]][state[2]][state[3]]
                self.b.append((1 - self.epsilon) + self.epsilon / self.actionsize)
            self.A.append(action)
            reward, state, done = self.env.takeaction(action)
            self.R.append(reward)
            self.S.append(state)
            if done:
                break

    def test(self):
        counter = 0
        state = self.env.reset()
        print("state", state)
        testS = []
        testS.append(state)
        while True:
            counter += 1
            action = self.PI[state[0]][state[1]][state[2]][state[3]]
            print("action:", action/ 3 - 1, action % 3 -1)
            reward, state, done = self.env.takeaction(action)
            print("reward, state, done", reward, state, done)
            testS.append(state)
            if done:
                print('done!')
                break
            if counter > 100:
                break
        print("text length:", len(testS))


    def train(self, stepnum):
        for step in range(0, stepnum):
            print("step:", step)
            self.genepisode()
            G = 0.0
            W = 1.0
            print("len", len(self.A))
            for t in range(len(self.A) - 1, -1, -1):
                #print("t", t)
                G = GAMMA * G + self.R[t + 1]
                s = self.S[t]
                action = self.A[t]
#                print("s, action:", s, action)
                self.C[s[0]][s[1]][s[2]][s[3]][action] += W
#                print(self.C[s[0]][s[1]][s[2]][s[3]][action])
#                print(self.Q[s[0]][s[1]][s[2]][s[3]][action])
                self.Q[s[0]][s[1]][s[2]][s[3]][action] += W / self.C[s[0]][s[1]][s[2]][s[3]][action] * (G - self.Q[s[0]][s[1]][s[2]][s[3]][action])
#                print(self.Q[s[0]][s[1]][s[2]][s[3]][action])
                self.updatePI()
                if action != self.PI[s[0]][s[1]][s[2]][s[3]]:
                    break
#                print("action", action)
                W = W / self.b[t]

    def printresult(self):
        print(self.Q)

    def printstartresult(self):
        for i in range(3, 8):
            print("column:", i)
            print(self.Q[self.row - 1][i][0][0])
            action = self.PI[self.row - 1][i][0][0]
            print("action:", action, action / 3 - 1, action % 3 - 1)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    env = Environment()
    agent = Agent(env.rowsize(), env.columnsize(), env.rspeedlimit(), env.cspeedlimit(), env.actionsize(), env)
    agent.train(10000)
#    agent.printresult()
    agent.printstartresult()
    agent.test()

#    agent.policy_eval()
#    agent.getpolicy()


if __name__ == "__main__":
    sys.exit(main())