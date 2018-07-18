import numpy as np
import random
import matplotlib.pyplot as plt

# constant step-size
ALPHA = 0.1
# exploration parameter
EPSILONINIT = 1.0/256
# number of steps in each episode
MAXSTEP = 20000
# start counting the average reward
STARTCOUNT = 10000
# try how many times
EPISODENUM = 7

AVGR = np.zeros(EPISODENUM)

PARA = np.zeros(EPISODENUM)

#average reward for constant step-size
ARF = np.zeros(MAXSTEP, float)
#optimal action for constant step-size
OptimalF = np.zeros(MAXSTEP, float)

# random walk parameters
MEAN = 0
STD = 0.01

# number of arms
K = 10

# initial Qtrue
Qinit = 1

Qtrue = np.zeros(K, float)

# init Qtrue value
def initQtrue(K):
    global Qtrue
    Qtrue = np.zeros(K, float)
    for i in range(0, K):
        Qtrue[i] = Qinit

# random walk after each step
def updateQtrue(K):
    global Qtrue
    for i in range(0, K):
        Qtrue[i] += random.gauss(MEAN, STD)

# get the best action
def bestaction():
    return np.argmax(Qtrue)

# take action A. return whether it is a best action and the reward
def bandit(A):
    return A == bestaction(), Qtrue[A] + np.random.randn()


QF = np.zeros(K, float)
print(QF)

MAXCOUNT = 100

for c in range(0, MAXCOUNT):
    print(c)
    EPSILON = EPSILONINIT
    for b in range(0, EPISODENUM):
        initQtrue(K)
        QF = np.zeros(K, float)
        AVG = 0
        for a in range(1, MAXSTEP):
            epsilon = random.random()
            if epsilon < EPSILON:
                AF = random.randint(0, K-1)
            else:
                AF = np.argmax(QF)

            bestRF, RF = bandit(AF)
            if bestRF:
                OptimalF[a] += 1

            QF[AF] = QF[AF] + ALPHA * (RF - QF[AF])

            updateQtrue(K)
            if a > STARTCOUNT:
                AVG = AVG + 1.0 / (a - STARTCOUNT) * (RF - AVG)
        PARA[b] = EPSILON
        AVGR[b] += AVG
        EPSILON = EPSILON * 2

AVGR = AVGR / MAXCOUNT

print(PARA)
print(AVGR)

f = plt.figure(0)
plt.plot(PARA, AVGR, label=' Constant Step-size ' + str(ALPHA))
plt.xlabel('Epsilon')
plt.ylabel('Average reward over 10000-20000 steps')
plt.legend()

f.savefig("avgreward.pdf")

plt.show()
