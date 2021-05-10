import numpy as np

def read_data(filename):
    # read header
    fin = open(filename, 'r')
    
    line = fin.readline()
    words = line.split()
    steps = int(words[3])
    
    line = fin.readline()
    words = line.split()
    t = float(words[3])
    
    line = fin.readline()
    words = line.split()
    g = float(words[3])
    
    fin.close()
    
    # read data
    data = np.genfromtxt(filename, skip_header = 3)
    
    x = data[:,0]
    b = data[:,1]
    s = data[:,2]
    u = data[:,3]
    
    return steps, t, g, x, b, s, u