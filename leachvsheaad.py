import numpy as np
import matplotlib.pyplot as plt

# Define the LEACH algorithm
def leach_algo(cluster_members, dead_nodes, p, round_number):
    N = len(cluster_members)
    threshold = p / (1 - p * (round_number % (1 / p)))

    is_cluster_head = np.zeros(N, dtype=bool)

    for i in range(N):
        if not dead_nodes[i] and np.random.rand() < threshold:
            is_cluster_head[i] = True

    return is_cluster_head, is_cluster_head.sum()

# Energy calculation function
def calculate_energy(E, CH, D, CHpl, NonCHpl, Etrans, Eagg, Efs, Erec, numClust, N):
    if numClust > 0:
        E[CH] = E[CH] - ((Etrans + Eagg) * CHpl + Efs * CHpl * (D ** 2) + NonCHpl * Erec * round(N / numClust))
    return E

# Initialize parameters
N = 100  # Number of nodes
W = 200  # Length of the network
L = 200  # Width of the network
Ei = 2  # Initial energy of each node (joules)
CHpl = 3000  # Packet size for cluster head per round (bits)

p = 0.05  # Desired percentage of cluster heads
R = 50  # Range for cluster
pMin = 10 ** -4  # Lowest possible CH_prop

num_rounds = 2000  # Max Number of simulated rounds
NonCHpl = 200  # Packet size for normal node per round (bits)
Tsetup = 4  # Average Time in seconds taken in setup phase
Tss = 10  # Average Time in seconds taken in steady-state phase
Etrans = 1.0000e-05  # Energy for transmitting one bit
Erec = 1.0000e-05  # Energy for receiving one bit
Eagg = 1.0000e-07  # Data aggregation energy
Efs = 0.3400e-9  # Energy of free space model amplifier

# Position of sink
SX, SY = W / 2, L / 2

# Initialize variables
net = np.zeros((3, N))  # 1st row: states of being a CH, 1: never been CH, 0: has been CH
net[1, :] = np.random.rand(N) * W  # 2nd: x-position
net[2, :] = np.random.rand(N) * L  # 3rd: y-position

# Preallocation for energy calculations
E = np.full(N, Ei)  # Energy left in each node
EH = np.zeros(num_rounds)
Ecrit = 0
Ncrit = int(0.95 * N)
Dead = np.zeros(N, dtype=bool)
DeadH = np.zeros(num_rounds)
BitsH = np.zeros(num_rounds)

# Simulating for each round
for r in range(num_rounds):
    # Choosing Cluster heads
    net[0, :], CH = leach_algo(net[0, :], Dead, p, r)
    tmp = np.where(CH)[0]
    
    for i in range(N):
        if tmp.size == 0:
            continue
        aa = np.argmin(np.sqrt((net[1, CH] - net[1, i]) ** 2 + (net[2, CH] - net[2, i]) ** 2))
        net[0, i] = tmp[aa]

    # Energy calculations
    EH[r] = np.sum(E)
    numClust = np.sum(CH)
    D = np.sqrt((net[1, CH] - SX) ** 2 + (net[2, CH] - SY) ** 2)

    # Calculate energy for cluster heads and non-cluster heads
    E = calculate_energy(E, CH, D, CHpl, NonCHpl, Etrans, Eagg, Efs, Erec, numClust, N)

    rest = N - numClust - np.sum(Dead)
    mD = np.zeros(rest)
    tmp = net[1:3, (~CH) & (~Dead)]

    for i in range(rest):
        mD[i] = np.min(np.sqrt((net[1, CH] - tmp[0, i]) ** 2 + (net[2, CH] - tmp[1, i]) ** 2))

    E[~CH & ~Dead] = calculate_energy(E[~CH & ~Dead], False, mD, CHpl, NonCHpl, Etrans, Eagg, Efs, Erec, numClust, N)

    E[Dead] = 0
    Dead[E <= Ecrit] = True
    DeadH[r] = np.sum(Dead)

    BitsH[r + 1] = BitsH[r] + numClust * CHpl + rest * NonCHpl

# Plotting analysis of network performance & comparison
T = (Tsetup + Tss) * np.arange(0, r)
EH = EH[:r]
EHdis = (N * Ei) - EH
DeadH = DeadH[:r]
AliveH = N - DeadH
BitsH = BitsH[1:r + 1]

plt.figure(figsize=(10, 6))
plt.plot(T, EHdis, '-x', label='LEACH Energy')
plt.xlabel('Time (s)')
plt.ylabel('Total Energy Dissipated (Joules)')
plt.title('LEACH Energy Dissipation Over Time')
plt.legend()
plt.grid(True)
plt.show()



# Define the HEED algorithm
def HEED_algo(R, Dead, p, pMin, E, Ei, net, cost):
    N = len(net)
    CH = np.zeros(N, dtype=bool)

    for i in range(N):
        if not Dead[i]:
            if np.random.rand() < p:
                CH[i] = True

    return CH, net

# Define the funH function (a basic placeholder, replace with your implementation)
def funH(x, y, net, CH, SX, SY):
    return np.sqrt((net[1, CH] - x) ** 2 + (net[2, CH] - y) ** 2).min()

N = 100
W = 200
L = 200
Ei = 2
CHpl = 3000

p = 0.05
R = 50
pMin = 10 ** -4

num_rounds = 2000
NonCHpl = 200
Tsetup = 4
Tss = 10
Etrans = 1.0000e-05
Erec = 1.0000e-05
Eagg = 1.0000e-07
Efs = 0.3400e-9

SX, SY = W / 2, L / 2

net = np.zeros((3, N))
net[1, :] = np.random.rand(N) * W
net[2, :] = np.random rand(N) * L

E = np.full(N, Ei)
EH = np.zeros(num_rounds)
Ecrit = 0
Ncrit = int((98 / 100) * N)
Dead = np.zeros(N, dtype=bool)
DeadH = np.zeros(num_rounds)
BitsH = np.zeros(num_rounds)

for r in range(num_rounds):
    CH, net = HEED_algo(R, Dead, p, pMin, E, Ei, net, cost)

    EH[r] = np.sum(E)
    numClust = np.sum(CH)
    D = np.sqrt((net[1, CH] - SX) ** 2 + (net[2, CH] - SY) ** 2)
    E[CH] = E[CH] - ((Etrans + Eagg) * CHpl + Efs * CHpl * (D ** 2) + NonCHpl * Erec * round(N / numClust))

    rest = N - numClust - np.sum(Dead)
    mD = np.zeros(rest)
    tmp = net[1:3, (~CH) & (~Dead)]
    for i in range(rest):
        mD[i] = funH(tmp[0, i], tmp[1, i], net, CH, SX, SY)

    E[~CH & ~Dead] = E[~CH & ~Dead] - (NonCHpl * Etrans + Efs * CHpl * (mD ** 2) + (Erec + Eagg) * CHpl)

    E[Dead] = 0
    CH[Dead] = False
    Dead[E <= Ecrit] = True
    DeadH[r] = np.sum(Dead)

    BitsH[r + 1] = BitsH[r] + numClust * CHpl + rest * NonCHpl

T = (Tsetup + Tss) * np.arange(0, r)
EH = EH[:r]
EHdis = (N * Ei) - EH
DeadH = DeadH[:r]
AliveH = N - DeadH
BitsH = BitsH[1:r + 1]

plt.figure(figsize=(10, 6))
plt.plot(T, EHdis, '-x')
plt.xlabel('Time (s)')
plt.ylabel('Energy Dissipated (Joules)')
plt.title('Total Energy Dissipated (HEED)')
plt.grid(True)
plt.show()
