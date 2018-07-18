import numpy as np
import random
import matplotlib.pyplot as plt

# constant step-size
ALPHA = 0.1
# exploration parameter
EPSILON = 0.1
# number of steps in each episode
MAXSTEP = 10000
# try how many times
EPISODENUM = 1000

#average reward for sample average
AR = np.zeros(MAXSTEP, float)
#optimal action for sample average
Optimal = np.zeros(MAXSTEP, float)

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


Q = np.zeros(K, float)
QF = np.zeros(K, float)
N = np.zeros(K, float)
print(Q)

for b in range(1, EPISODENUM):
    print(b)
    initQtrue(K)
    for a in range(1, MAXSTEP):
        epsilon = random.random()
        if epsilon < EPSILON:
            A = random.randint(0, K-1)
            AF = random.randint(0, K-1)
        else:
            A = np.argmax(Q)
            AF = np.argmax(QF)

        bestR, R = bandit(A)
        if bestR:
            Optimal[a] += 1
        AR[a] += R

        bestRF, RF = bandit(AF)
        if bestRF:
            OptimalF[a] += 1
        ARF[a] += RF

        N[A] = N[A] + 1
        Q[A] = Q[A] + 1 / N[A] * (R - Q[A])
        QF[A] = QF[A] + ALPHA * (R - QF[A])

        updateQtrue(K)


Optimal = Optimal / EPISODENUM
OptimalF = OptimalF / EPISODENUM
AR = AR / EPISODENUM
ARF = ARF / EPISODENUM

f = plt.figure(0)
plt.plot(AR, label='epsilon = ' + str(EPSILON) + ' Sample Average')
plt.plot(ARF, label='epsilon = ' + str(EPSILON) + ' Constant Step-size ' + str(ALPHA))
plt.xlabel('Steps')
plt.ylabel('average reward')
plt.legend()

f.savefig("avgreward.pdf")

f = plt.figure(1)
plt.plot(Optimal, label='epsilon = ' + str(EPSILON) + ' Sample Average')
plt.plot(OptimalF, label='epsilon = ' + str(EPSILON) + ' Constant Step-size ' + str(ALPHA))
plt.xlabel('Steps')
plt.ylabel('% optimal action')
plt.legend()

f.savefig("optimal.pdf")

plt.show()

