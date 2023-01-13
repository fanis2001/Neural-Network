import matplotlib.pyplot as plt

def squad_error(file_name):
    with open(file_name, "r") as f:
        lines = f.readlines()

    x = []
    y = []
    for line in lines:
        x_coord, y_coord = line.split()
        x.append(float(x_coord))
        y.append(float(y_coord))

    plt.plot(x, y)
    
    plt.title("SQUAD ERROR")
    
    plt.show()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def draw_points(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    x = []
    y = []
    c = []
    p = []
    for line in lines:
        x_coord, y_coord, c_val, p_val = line.split()
        x.append(float(x_coord))
        y.append(float(y_coord))
        c.append(c_val)
        p.append(p_val)

    for i in range(len(x)):
        if c[i] == 'C1':
            color = 'green'
        elif c[i] == 'C2':
            color = 'blue'
        elif c[i] == 'C3':
            color = 'red'
        
        if p[i] == '+':
            marker = '+'
        elif p[i] == '-':
            marker = '*'
            color = "black"
        else:
            continue
        plt.scatter(x[i], y[i], marker=marker, color=color)

    plt.show()

squad_error("squad_error_per_epoch.txt")
draw_points("test.txt")