import matplotlib.pyplot as plt
import numpy as np

def houghLines(edged,rho_res,theta_res,thresholdVotes,filterMultiple,thresholdPixels=0):
    
    rows, columns = edged.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0/theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta)-2::-1]))
    
    #defining empty Matrix in Hough space, where x is for theta and y is x*cos(theta)+y*sin(theta)
    diagonal = np.sqrt((rows - 1)**2 + (columns - 1)**2)
    q = np.ceil(diagonal/rho_res)
    nrho = 2*q + 1
    rho = np.linspace(-q*rho_res, q*rho_res, nrho)
    houghMatrix = np.zeros((len(rho), len(theta)))
    
    #Here we populate houghMatrix
    for rowId in range(rows):                               #for each x in edged
        for colId in range(columns):                        #for each y in edged
          if edged[rowId, colId]>thresholdPixels:           #edged has values 0 or 255 in our case
            #for each theta we calculate rhoVal, then locate it in Hough space plane
            for thId in range(len(theta)):
              rhoVal = colId*np.cos(theta[thId]*np.pi/180.0) + \
                  rowId*np.sin(theta[thId]*np.pi/180)
              rhoIdx = np.nonzero(np.abs(rho-rhoVal) == np.min(np.abs(rho-rhoVal)))[0]
              houghMatrix[rhoIdx[0], thId] += 1  
            
   
   #cluster and filter multiple dots in Houghs plane
    if filterMultiple>0:
        clusterDiameter=filterMultiple
        values=np.transpose(np.array(np.nonzero(houghMatrix>thresholdVotes)))
        filterArray=[]
        filterArray.append(0)
        totalArray=[]
        for i in range (0, len(values)):
            if i in filterArray[1::]:
                continue
            tempArray=[i]
            for j in range (i+1, len(values)):
                if j in filterArray[1::]:
                    continue
                for k in range (0, len(tempArray)):
                    if getLength(values[tempArray[k]],values[j])<clusterDiameter:
                        filterArray.append(j)
                        tempArray.append(j)
                        break
            totalArray.append(tempArray)
        
        #leave the highest value in each cluster
        for i in range (0, len(totalArray)):
             for j in range (0, len(totalArray[i])):
                 if j==0:
                     highest=houghMatrix[values[totalArray[i][j]][0],values[totalArray[i][j]][1]]
                     ii=i
                     jj=j
                 else:
                     if houghMatrix[values[totalArray[i][j]][0],values[totalArray[i][j]][1]]>=highest:
                         highest=houghMatrix[values[totalArray[i][j]][0],values[totalArray[i][j]][1]]
                         houghMatrix[values[totalArray[ii][jj]][0],values[totalArray[ii][jj]][1]]=0
                         ii=i
                         jj=j
                     else:
                         houghMatrix[values[totalArray[i][j]][0],values[totalArray[i][j]][1]]=0
                    
    return (np.where(houghMatrix>thresholdVotes)[0]-q)*rho_res, theta[np.where(houghMatrix>thresholdVotes)[1]]*np.pi/180.0
    
def getLength(startPoint,secondPoint):
    ##Inputs:
    #startPoint - [x,y]
    #secondPoint - [x,y]

    ##Outputs:
    #lenv - length between two points

    v1x=secondPoint[0]-startPoint[0]
    v1y=secondPoint[1]-startPoint[1]
    lenv=np.sqrt(v1x*v1x+v1y*v1y)
    return lenv
    
def unique(a):
    ##Inputs:
    #a - list of 1xN arrays

    ##Outputs:
    #b - filtered array

    #Example
    # a=[array([ 1,  3, 12, 17]),
    #    array([ 1,  3, 17, 12]),
    #    array([ 1,  3, 18, 20])]
    # b -> [array([ 1,  3, 12, 17]),
    #       array([ 1,  3, 18, 20])]


    b=np.array(a)
    a=np.sort(np.array(a))
    order = np.lexsort(a.T)
    a = a[order]
    b = b[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    return b[ui]
    
def reorderPoints(corners):
    ##Inputs:
    #corners - list of corners (look at example)

    ##Outputs:
    #array - reordered corners array

    #Example
    #corenrs=[[153, 104], [255, 98], [178, 144], [231, 58]]
    #array -> [[153, 104], [178, 144], [255, 98], [231, 58]]
    
    array=[]
    for i in range (0, len(corners)):
        tempArray=[]
        length1=getLength(corners[i][0],corners[i][1])
        length2=getLength(corners[i][0],corners[i][2])
        length3=getLength(corners[i][0],corners[i][3])
        lenArr=np.array([length1,length2,length3])
        tempArray.append(corners[i][0])
        tempArray.append(corners[i][1+np.where(np.array(lenArr)==np.min(lenArr))[0][0]])
        lenArr[np.where(np.array(lenArr)==np.min(lenArr))[0][0]]+=-0.00001 #n case of rectangle
        tempArray.append(corners[i][1+np.where(np.array(lenArr)==np.max(lenArr))[0][0]])
        tempArray.append(corners[i][1+np.where(np.array(lenArr)==np.median(lenArr))[0][0]])
        array.append(tempArray)
    return array
    
def getAngle(startPoint,secondPoint,thirdPoint, absol=True):
    #Gets angle between vectors (startPoint,secondPoint) and vector
    #(secondPoint,thirdPoint)
    
    ##Inputs:
    #startPoint - [x,y]
    #secondPoint - [x,y]
    #thirdPoint - [x,y]

    ##Outputs:
    #angle - angle between two vectors

    
    v1x=secondPoint[0]-startPoint[0]
    v1y=secondPoint[1]-startPoint[1]
    v2x=thirdPoint[0]-startPoint[0]
    v2y=thirdPoint[1]-startPoint[1]
    
    lenv1=np.sqrt(v1x*v1x+v1y*v1y)
    lenv2=np.sqrt(v2x*v2x+v2y*v2y)
    
    angle=np.arccos((v1x*v2x+v1y*v2y)/(lenv1*lenv2))
    
    a=1
    if absol==False:
        a = np.sign((v1x) * (v2y) - (v1y) * (v2x))
    
    if np.absolute(angle) < 0.02:
        angle=0
    return a*angle
    
def plotHoughLines(rho,theta,image):
  
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho

    fig2, ax1 = plt.subplots(ncols=1, nrows=1)
    ax1.imshow(image)
    
    for i in range (0, len(rho)):   
        ax1.plot( [x0[i] + 1000*(-b[i]), x0[i] - 1000*(-b[i])],
                  [y0[i] + 1000*(a[i]), y0[i] - 1000*(a[i])], 
                  'xb-',linewidth=3)
    
    ax1.set_ylim([image.shape[0],0])
    ax1.set_xlim([0,image.shape[1]])
    
    plt.show()

