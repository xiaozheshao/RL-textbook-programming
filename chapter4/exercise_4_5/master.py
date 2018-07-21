import numpy as np
import math
import sys

MAXNUM = 20
theta = 0.00001
gamma = 0.9
ACTIONNUM = 5
PARKINGSIZE = 10

class Agent(object):
    def __init__(self):
        self.V = np.zeros((MAXNUM + 1, MAXNUM + 1), dtype =float)
        self.Vtmp = np.zeros((MAXNUM + 1, MAXNUM + 1), dtype=float)
        self.PI = np.zeros((MAXNUM + 1, MAXNUM + 1), dtype=int)
        self.matrix = np.zeros((MAXNUM + 1, MAXNUM + 1, MAXNUM + 1, MAXNUM + 1, ACTIONNUM * 2 + 1), dtype = float)
        self.rmatrix = np.zeros((MAXNUM + 1, MAXNUM + 1, MAXNUM + 1, MAXNUM + 1, ACTIONNUM * 2 + 1), dtype = float)
        self.initmatrix()
        print("initialize")

    def initmatrix(self):
        for aa in range(0, MAXNUM + 1):
            print("aa", aa)
            for bb in range(0, MAXNUM + 1):
                for a in range(0, MAXNUM + 1):
                    for b in range(0, MAXNUM + 1):
                        for action in range(0, ACTIONNUM * 2 + 1):
                            self.matrix[aa][bb][a][b][action], self.rmatrix[aa][bb][a][b][action] = self.statetransfer(aa, bb, a, b, action)
#        print(self.matrix)
        print("******************************************************************")
#        print(self.rmatrix)
        sum = self.check()
        self.weight(sum)
        print("******************************************************************************************************************")
        self.check()

    def weight(self, sum):
        for a in range(0, MAXNUM + 1):
            for b in range(0, MAXNUM + 1):
                for action in range(0, ACTIONNUM * 2 + 1):
                    for aa in range(0, MAXNUM + 1):
                        for bb in range(0, MAXNUM + 1):
                            self.matrix[aa][bb][a][b][action] /= sum[a][b][action]

    def check(self):
        sum = np.zeros((MAXNUM + 1, MAXNUM + 1, ACTIONNUM * 2 + 1))
        for a in range(0, MAXNUM + 1):
            for b in range(0, MAXNUM + 1):
                for action in range(0, ACTIONNUM * 2 + 1):
                    for aa in range(0, MAXNUM + 1):
                        for bb in range(0, MAXNUM + 1):
                            sum[a][b][action] += self.matrix[aa][bb][a][b][action]
                   # print("a, b, action, sum:", a, b, action, sum[a][b][action])
        return sum

    def accumulatepoison(self, lamb, n):
        paccu = 0
        for tmp in range(0, n):
            paccu += float(lamb ** tmp) / math.factorial(tmp) * math.exp(-lamb)
        return paccu

    def statetransfer(self, aa, bb, a, b, action):
        action = action - ACTIONNUM
        if action > 0:
            transnum = min(action, a)
        elif action < 0:
            transnum = max(action, -b)
        else:
            transnum = 0
        a = a - transnum
        b = b + transnum
        r = -2 * abs(action)
        if a > MAXNUM:
            a = MAXNUM
        if b > MAXNUM:
            b = MAXNUM

        #additional conditions
        if action > 0:
            r += 2
        if a > PARKINGSIZE:
            r -= 4
        if b > PARKINGSIZE:
            r -= 4

        p = 0
        income = 0
        for n in range(0, a + 1):
            if n == a:
                pn = 1 - self.accumulatepoison(3, n)
            else:
                pn = (3.0 ** n) / math.factorial(n) * math.exp(-3)
            assert pn >= 0
            returna = aa - (a - n)
            if returna < 0:
                preturna = 0
                continue
            elif returna == MAXNUM:
                preturna = 1 - self.accumulatepoison(3, returna)
            else:
                preturna = (3.0 ** returna) / math.factorial(returna) * math.exp(-3)
            assert preturna >= 0
            for m in range(0, b + 1):
                if m == b:
                    pm = 1 - self.accumulatepoison(4, m)
                else:
                    pm = (4.0 ** m) / math.factorial(m) * math.exp(-4)
                assert pm >= 0
                returnb = bb - (b - m)
                if returnb < 0:
                    preturnb = 0
                    continue
                elif returnb == MAXNUM:
                    preturnb = 1 - self.accumulatepoison(2, returnb)
                else:
                    preturnb = (2.0 ** returnb) / math.factorial(returnb) * math.exp(-2)
                assert returnb >= 0
                ptmp = pn * pm * preturna * preturnb
                assert ptmp >= 0
                p += ptmp
                income += (n + m) * 10 * ptmp
        if p != 0:
            income = income / p
            r += income
        return p, r


    def updateV(self, a, b, action):
        rst = 0
        for bb in range(0, MAXNUM + 1):
            for aa in range(0, MAXNUM + 1):
#                p, r = self.statetransfer(aa, bb, a, b, action)
                p = self.matrix[aa][bb][a][b][action]
                r = self.rmatrix[aa][bb][a][b][action]
                rst += p * (r + gamma * self.V[aa][bb])
        return rst


    def policy_eval(self):
        while True:
            delta = 0
            for a in range(0, MAXNUM + 1):
                for b in range(0, MAXNUM + 1):
                    v = self.V[a][b]
                    self.V[a][b] = self.updateV(a,b, self.PI[a][b] + ACTIONNUM)
                    delta = max(delta, abs(v - self.V[a][b]))
#            self.V = self.Vtmp
#            self.printresult()
            if delta < theta:
                break

    def policy_improve(self):
        stable = True
        for a in range(0, MAXNUM + 1):
            for b in range(0, MAXNUM + 1):
                oldaction = self.PI[a][b]
                actions = np.zeros(ACTIONNUM * 2 + 1)
                for action in range(0, ACTIONNUM * 2 + 1):
                    actions[action] = self.updateV(a,b, action)
                self.PI[a][b] = np.argmax(actions) - ACTIONNUM
                if oldaction != self.PI[a][b]:
                    stable = False
        if stable:
            return False
        else:
            return True

    def printresult(self):
        print("-----------------------------")
        print(self.V)
        print(self.PI)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    agent = Agent()
    agent.policy_eval()
    while agent.policy_improve():
#        agent.printresult()
        agent.policy_eval()
    agent.printresult()


if __name__ == "__main__":
    sys.exit(main())