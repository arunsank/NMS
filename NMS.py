import os
import sys
import argparse
import numpy as np
from numpy import genfromtxt


def iouFilter(box,
              boxArea, 
              otherBoxes,
              otherboxesArea,
              threshold):
    """
    Calculate Iou and return indexes of low confidence
    boxes that need to be suppressed
      
    """
    
    x1 = np.maximum(box[0],otherBoxes[:,0])
    y1 = np.maximum(box[1],otherBoxes[:,1])
    x2 = np.minimum((box[0]+box[2]),(otherBoxes[:,0]+otherBoxes[:,2]))
    y2 = np.minimum((box[1]+box[3]),(otherBoxes[:,1]+otherBoxes[:,3]))
    
    #Calculate Area of intersection
    intersectionArea =np.maximum(y2-y1,0)* np.maximum(x2-x1,0)
    #Calculate iou  ( IoU = Area of intersection /Area of Union)
    ioU = intersectionArea/(boxArea+otherboxesArea- intersectionArea)
    return list(np.where(ioU>threshold)[0])


def NMS(boxes,threshold = 0.5):
    """
    Perform Non maximal suppression on bounding boxes and return them
    Input
    boxes - Input bounding boxes
    threshold - Iou thresold level
    Output
    Final bounding boxes after non max suppression
    """
    if len(boxes)<2:
        return boxes
    # Extract width and height of bounding boxes
    W= boxes[:,2]
    H = boxes[:,3]
    boxArea = np.multiply((W+1),(H+1))
    #Sort boxes by confidence scores to choose most confident boxes first
    orderedBoxes = list(boxes[:,4].argsort())
    finalboxIx = []
    
    #Loop through confidence sorted boxes
    while orderedBoxes:
        currIx = orderedBoxes.pop()
        finalboxIx.append(currIx)
        if not len(orderedBoxes):
            break
        #Perform Iou threshold to suppress boxes that overlap more with current bounding box
        suppressboxesIx = iouFilter(boxes[currIx],boxArea[currIx], boxes[orderedBoxes],boxArea[orderedBoxes],threshold)
        orderedBoxes = list(set(orderedBoxes) - set([orderedBoxes[ix] for ix in suppressboxesIx]))
        
    return boxes[finalboxIx]


if __name__ == '__main__':
    #Parse input arguments
    inpArgs = argparse.ArgumentParser()
    inpArgs.add_argument("--inputfilePath", required=True,
    help="path to input files with bounding boxes")
    inpArgs.add_argument("--inputfileName", required=True,
    help="path to input file")
    inpArgs.add_argument("--iou", required=True,
    help="iou threshold for the boxes")
    inpArgs.add_argument("--outputfilePath", required=False,
    help="path to output file")
    
    args = inpArgs.parse_args()
    if not os.path.isdir(args.inputfilePath):
        print ("Invalid input image path")
        sys.exit()
    if not os.path.isfile(args.inputfilePath+'/'+ str(args.inputfileName)):
        print ("Input file doesnot exist")
        sys.exit()
    inputPath = args.inputfilePath
    inputfileName= args.inputfileName
    iouThreshold = float(args.iou)
    if args.outputfilePath:
        if not os.path.isdir(args.outputfilePath):
            print ("Invalid output file path")
            sys.exit()
        outputfilePath = args.outputfilePath
    else:
        outputfilePath = '.'
    
    #Read input file   
    boxes = genfromtxt(inputPath+ '/' + inputfileName, delimiter=',')
    if not len(boxes):
        print ("Input File empty")
        sys.exit()
    #Run Non maximal Suppression on input
    finalBoxes = NMS(boxes, iouThreshold)
    #Output file with final bounding boxes
    np.savetxt(outputfilePath+'/'+ inputfileName[:-4]+'Output.csv',finalBoxes, fmt="%.2f",delimiter =',')
    

