import os
import numpy as np
import time

start = time.time()
lilos = ['M81R','R12','R18','R1','R1twin','R3','R4','R67','R6','Rluo']
n=3
for i in range(500):
    lilo=np.random.choice(lilos)
    freq = np.random.randint(50, 300, 1)[0]
    timestep = np.random.randint(1,100,1)[0]/10000
    dm = np.random.randint(50,2950,1)[0]
    loc = np.random.randint(0,600,1)[0]-300

    low_high=np.random.randint(1,4,1)[0]
    if low_high<3:
        snr = np.random.randint(65,100,1)[0]/100
    else:
        snr = np.random.randint(100,1000,1)[0]/100

    scint_test = np.random.randint(1,15)
    if scint_test<3:
        scints = 2
    elif scint_test>12:
        scints = 1
    else:
        scints = 0

    print("python -f "+str(freq)+" -t "+str(timestep)+" -d "+str(dm)+" -n "+lilo+" -l "+str(np.array(loc))+ " -snr "+str(snr)+" -scints "+str(int(scints)))

    os.system("python param_test_heim_fetch.py -f "+str(freq)+" -t "+str(timestep)+" -d "+str(dm)+" -n "+lilo+" -l "+str(np.array(loc))+ " -snr "+str(snr)+" -scints "+str(int(scints)))

end = time.time()

print("The running of {} Dm values took {} seconds".format(n,end-start))
