'''

3-dimensional plotter with 2-channel color input

Created by Jackson Banbury as part of the EA302 SWIM Team at the University of Toronto

March, 2018

Created for use with the Deltyburn 3D printer collecting waveform
data from an array of transducers, using a lock-in amplifier for recording


'''
#Use 'Agg' for no-display backend (I do this so I can run it via SSH, but it means no preview)
import matplotlib
matplotlib.use('Agg')

import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import scipy.stats as stats
from sklearn.preprocessing import normalize
from colormap import rgb2hex



# Color scaling for compensation:
Rscale = 400
Gscale = 400
Bscale = 400

# Color buffer below which the points are not plotted
blackbuffer = 1

def readtxt(length):
    txtdata = [line.rstrip() for line in open('out5.txt')]

    return txtdata[-l:]


def unique(list1):
    # function to get unique layers from z

    unique_list = []

    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

def histogram(a):
    #Plots a histogram - I used this to check out the lockin data

    density = stats.gaussian_kde(a)
    fig = plt.hist(a, bins=np.linspace(0,max(a), 20), histtype=u'step', normed=True)

    plt.savefig('histogram.png')
    plt.close(fig)

def dual_normalizer(a, b):
    #This is how the alpha values are calculated based on an input of Re and Im

    #Normalized magnitude, multiplied by 255 in this case to reflect RGBA spectrum
    max_a = max(a[:])
    max_b = max(b[:])

    normalizer = max(max_a,max_b)

    normalized = []

    #This shift pushes the values to the left on the sigmoid function,
    #effectively decreasing alpha (and more significantly so for those close to the steep part of the curve)
    shift = 10

    for i in range(len(a))
        #Simple magnitude calculation between the Re and Im
        normal_magnitude = (a[i]**2 + b[i]**2)**0.5/normalizer

        #The parameterized logistic function (a type of Sigmoid function)
        #Recall the logistic function is: f(x) = (1/(1+e^(-x)))
        logistic_fxn = (1/(1+math.exp(-normal_magnitude*4-shift)))

        normalized.append(int(100*logistic_fxn*normal_magnitude))

    return normalized

def scatter3d(x,y,z, cs, Re, Im, colorsMap='jet'):
    
    #Change the size of the points to be plotted here (I use 10 mostly)
    pointsize = 10

    #Get the alpha values from the above function
    alpha = dual_normalizer(Re,Im)

    print('Done Normalizing')

    cm = plt.get_cmap(colorsMap)
    #cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    #scalarMap = cmx.ScalarMappable(norm=1, cmap=cm)

    fig = plt.figure()
    ax = Axes3D(fig)
    #ax.set_facecolor('xkcd:black')
    #fig, ax = plt.subplots()
    for i in range(len(x)):
        #Only plots non-black points
        #if not(cs[i] == cs[1]):
            #Only plot every 5th point
        if not(y[i]<0):
            temp = cs[i]
            frontcs = temp[:1]
            endcs = temp[1:]
            alpha_hex = '{a:02x}'.format(a=alpha[i])
            endcs += alpha_hex
            #print(alpha_hex)
            #cs[i] += '80'
            this_cs = frontcs + endcs
            #print(this_cs)
            ax.scatter(x[i], y[i], z[i], s=pointsize, color=this_cs)
    #scalarMap.set_array(cs)
    #fig.colorbar(scalarMap)

    ax.set_ylim(bottom=-70, top=70, auto=False)

    figname = "data_adc.png"
    fig.savefig(figname)
    plt.close(fig)

def scatter3d_slices_vertical(x,y,z, cs, Re, Im,  colorsMap='jet'):

    #Change the size of the points to be plotted here:
    pointsize = 30
    
    #Change the desired number of slices here:
    n_slices = 10
    
    cm = plt.get_cmap(colorsMap)


    #Combine all of the data into a single array for sorting
    data_arr = np.column_stack((x,y,z,cs))

    #Sort the whole array based on values in the Y column
    data_arr = data_arr[data_arr[:,1].argsort()]

    #Must convert the data back into its original format
    #(Numpy likes to amalgamate the array to one type - in this case string)
    x = data_arr[:,0].astype(float)
    y = data_arr[:,1].astype(float)
    z = data_arr[:,2].astype(float)
    cs = data_arr[:,3].astype(str)
    
    img_counter = 0

    ylist = []

    #Gets a chunked list of y values
    for i in range(0, len(y), n_slices):
        ylist.append(y[i:i+n_slices])

    #print(ylist)
    for chunk in range(n_slices):
        #print(ylist[chunk])
        fig = plt.figure()
        ax = Axes3D(fig)
        for layer in ylist[chunk]:
            for i in range(len(y)):
                #Only plots non-black points
                if(True):
                    #take only the points for the current layer

                    #The try/except is necessary due to the fact that 'layer' could be a single
                    #float value for y, for example at the ends of the cylinder,
                    #and you can't iterate a single float value with 'in'..
                    try:
                        if(y[i] in layer):
                            #This is a placeholder in case you want to plot every 5th point or something
                            #if not(i%5):  <-- this is an example of taking every fifth point
                            if(True):
                                ax.scatter(x[i], y[i], z[i], s=pointsize, color=cs[i])
                    except:
                        if(y[i] == layer):
                            if(True):
                                ax.scatter(x[i], y[i], z[i], s=pointsize, color=cs[i])

        #Sets a hard limit on the x and y axes so that it does not distort the slices
        ax.set_ylim(bottom=-70, top=70, auto=False)
        ax.set_xlim(left=-70, right=70, auto=False)

        #Change the name of the figure to be saved below:
        figname = "data_adc_layer%d.svg" %(img_counter)

        fig.savefig(figname)
        plt.close(fig)

        #Prints the progress in terms of layers
        print("Saved layer %d of %d" %(img_counter+1, n_slices))
        img_counter +=1


def scatter3d_slices(x,y,z, cs, colorsMap='jet'):

    cm = plt.get_cmap(colorsMap)
    #cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    #scalarMap = cmx.ScalarMappable(norm=1, cmap=cm)

    #Get the unique layers from Z (no repeats)
    zlist = unique(z)
    print(zlist)
    for layer in zlist[:1]:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_facecolor('xkcd:black')
        #fig, ax = plt.subplots()
        for i in range(len(x)):
            #Only plots non-black points
            if not(cs[i] == cs[1]):
                #take only the points for the current layer
                if(z[i]==layer):
                    #Only plot every 5th point
                    if not(i%5):
                        temp = cs[i]
                        #print(temp)
                        #print(".")
                        frontcs = temp[:1]
                        endcs = temp[1:]
                        endcs += '0D'
                        #cs[i] += '80'
                        this_cs = frontcs + endcs
                        #print(this_cs)
                        ax.scatter(x[i], y[i], z[i], s=15, color=this_cs)
        #scalarMap.set_array(cs)
        #fig.colorbar(scalarMap)

        ax.set_ylim(bottom=-70, top=70, auto=False)

        figname = "data_adc_layer%s.svg" %(layer)
        fig.savefig(figname)
        plt.close(fig)

# Note that this assumes your CSV is already filled with columns 1,2,3,4,5
# cooresponding to X,Y,Z,Re,Im respectively.
#The file name should be changed below as well
file = open('data_adc.csv','r')
csvdata = file.readlines()
csvdata.pop(0)
Z = []
X = []
Y = []
Re = []
Im = []

#Strips the CSV to separate arrays
for datapoint in csvdata:
    x, y, z, re, im= datapoint.strip('\r\n').split(',')

    Z.append(float(z))
    X.append(float(x))
    Y.append(float(y))
    Re.append(float(re))
    Im.append(float(im))

l = 2*len(Z)

R = []
G = []
B = []

for i in range(len(Re)):
    re = Re[i]
    im = Im[i]
    # RED components from positive inputs
    if(re>=0):
        red1 = re
    else:
        red1 = 0
    # Because red is a component of yellow..
    if(im>=0):
        red2 = im
    else:
        red2 = 0

    red = ((red1+red2)/2)

    # BLUE components of negative imaginary
    if(im<0):
        blu = 0-im
    else:
        blu = 0

    # GREEN components - from postitive imaginary and negative real
    if(re<0):
        grn1 = 0-re
    else:
        grn1 = 0
    if(im>=2.5):
        grn2 = im
    else:
        grn2 = 0

    grn = ((grn1+grn2)/2)

    #Append the colors into their arrays. Change the scale with the parameters above
    R.append(min(int(red*Rscale),255))
    G.append(min(int(grn*Gscale),255))
    B.append(min(int(blu*Bscale),255))

#These colors are cast into hex form for Axes3D
colors = np.array((R,G,B), dtype = float)
hexcolors = []
for i in range(len(R)):
    if (R[i]+G[i]+B[i] > blackbuffer):
        hexcolors.append(rgb2hex(R[i],G[i],B[i]))
    # Any colors that are close to black are set to absolute black to be filtered later
    # Change the blackbuffer parameter above to adjust this (set to 0 to stop this behavior)
    else:
        hexcolors.append(rgb2hex(0,0,0))

#Call the desired function below:

#For a full 3D scatterplot, use this:
#scatter3d(X,Y,Z,hexcolors,Re,Im)

#For vertical slices (along Y axis), use this:
scatter3d_slices_vertical(X,Y,Z,hexcolors,Re,Im)
