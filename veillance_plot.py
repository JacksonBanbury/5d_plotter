'''

Veillance Plotter
Plots the veillance flux data obtained from the Deltyburn experiment

Written by Jackson Banbury with the EA302 lab team
April, 2018

Use with "python3 swim_plot.py filename_in.csv filename_out.png"

'''
#Use the below two lines if you want to run the script via SSH
#import matplotlib
#matplotlib.use('Agg')
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
from colormap import rgb2hex
import sys
import progressbar

def normalize_255(a):
    print("Normalizing colors...")
    a[:] = [x / max(a) for x in a]
    a[:] = [int(x*255) for x in a]
    print("Done normalizing colors")
    return(a)

def swim_plot(x,y,z,colors):
    #Change the size of the points plotted here
    pointsize=20

    #Change this to plot every x points
    #Example: set as 5 to plot every 5th point, or set as 1 to plot every point
    plot_every_x = 2

    #Change the color threshold here (0-255 scale)
    #points with color below this value will not be plotted
    color_threshold = 20

    #Set up the progress bar
    bar = progressbar.ProgressBar().start()
    bar.maxval = 101

    if(sys.argv[2]):
        figname = sys.argv[2]
    else:
        figname = "swim_plot.png"

    fig = plt.figure()
    ax = Axes3D(fig)

    #Normalize the colors to range(0-255)
    #Note that for efficiency only the points being used based on the plot_every_x
    #parameter are sent for normalization
    colors = normalize_255(colors[::plot_every_x])

    print("Plotting figure...")

    if not(plot_every_x == 1):
        print('Note: plotting one in every %d points'%plot_every_x)
        print("To change this, edit the plot_every_x parameter")

    one_percent_progress = int(len(y)/100)
    bar_val = 0
    j=0


    for i in range(len(y))[::plot_every_x]:
        if(True):
            if(colors[j]>color_threshold):
                hexcolor = rgb2hex(0,int(colors[j]),0)
                ax.scatter(x[i],y[i],z[i],s=pointsize,color=hexcolor)
            j+=1
        if not(i%one_percent_progress):
            bar_val += 1
            bar.update(bar_val)

    plt.show()
    fig.savefig(figname)
    print("Figure saved as %s")%figname
    plt.close(fig)


def read_csv(filename):
    file = open(filename, 'r')
    csvdata = file.readlines()
    csvdata.pop(0)
    X=[]
    Y=[]
    Z=[]
    Colors=[]

    for datapoint in csvdata:
        x, y, z, color = datapoint.strip('\r\n').split(',')[:4]
        X.append(float(x))
        Y.append(float(y))
        Z.append(float(z))
        Colors.append(float(color))

    return(X,Y,Z,Colors)

def main():
    try:
        X,Y,Z,Colors = read_csv(sys.argv[1])
    except:
        if(sys.argv[1]):
            print("Filename not found. Ensure you either have the full file path \
                    specified or that the file is located in the same working directory")
        else:
            print("Error: no .csv file name or path specified.\nUse 'python3 swim_plot.py \
                    filename_in.csv filename_out.png'")
        sys.exit()
    swim_plot(X,Y,Z,Colors)

if __name__ == '__main__':
    main()
