import sys
import numpy as np
import matplotlib.pyplot as plt


THETA = 0.0001

WINPERCENT = 0.5
GAMMA = 1

class Agent(object):
    def __init__(self):
        self.V= np.zeros(101, dtype= float)
        self.V[100] = 1
        self.PI = np.zeros(100, dtype=int)
        self.figvalue = plt.figure(0)

    def updateV(self, s, a):
        wins = s + a
        losts = s - a
        #print ("s", s, "a", a)
        winreward = WINPERCENT * (GAMMA * self.V[wins])
        lostreward = (1 - WINPERCENT) * (GAMMA * self.V[losts])
        return winreward + lostreward

    def getactions(self, s):
        actions = np.zeros(51, dtype=float)
        upperbound = min(s, 100 - s)
        for a in range(0, upperbound + 1):
            actions[a] = self.updateV(s, a)
        #actions[a] = 0
        return actions

    def policy_eval(self):
        counter = 0
        while True:
            delta = 0
            for s in range(1, 100):
                v = self.V[s]
                actions = self.getactions(s)
                self.V[s] = max(actions)
                delta = max(delta, abs(v - self.V[s]))
            self.plotvalue(counter)
            if delta < THETA:
                break
            counter += 1
        self.plotshow()

    def getpolicy(self):
        for s in range(1, 99):
            actions = self.getactions(s)
            actions[0] = 0
            self.PI[s] = np.argmax(actions)
        self.policyshow()

    def plotshow(self):
#        plt.show()
        self.figvalue.savefig("value_estimate.pdf")

    def plotvalue(self, index):
        plt.figure(0)
        plt.plot(self.V, label=' Step ' + str(index))
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
#        plt.legend()

    def policyshow(self):
        self.figpolicy = plt.figure(1)
        plt.plot(self.PI, label= ' policy ')
#        states = np.arange(100)
#        plt.scatter(states, self.PI)
        plt.xlabel(' Capital ')
        plt.ylabel('Final policy(stake)')
        plt.legend()
        plt.show()
        self.figpolicy.savefig("policy.pdf")


def main(argv=None):
    if argv is None:
        argv = sys.argv
    agent = Agent()
    agent.policy_eval()
    agent.getpolicy()


if __name__ == "__main__":
    sys.exit(main())