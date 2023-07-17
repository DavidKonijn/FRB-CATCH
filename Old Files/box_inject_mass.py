import os
import numpy as np
import time

start = time.time()
lilos = ['M81R','R18','R1','R1twin','R3','R4','R67','Rluo']
n=10

for i in range(n):
    lilo=lilos[np.random.randint(0, 8, 1)[0]]
    freq = np.random.randint(30, 120, 1)[0]
    timestep = np.random.randint(5,30,1)[0]/10000
    dm = np.random.randint(50,500,1)[0]
    loc = np.random.randint(0,100,1)[0]-50
    scints = 0

    print("python -f "+str(freq)+" -t "+str(timestep)+" -d "+str(dm)+" -n "+lilo+" -l "+str(np.array(loc))+ " -scints "+str(int(scints))+ " -id "+str(i))

    os.system("python box_inject_burst.py -f "+str(freq)+" -t "+str(timestep)+" -d "+str(dm)+" -n "+lilo+" -l "+str(np.array(loc))+ " -scints "+str(int(scints))+ " -id "+str(i))

end = time.time()

print("The running of {} Dm values took {} seconds".format(n,end-start))