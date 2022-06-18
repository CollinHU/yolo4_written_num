
import cv2
import numpy as np


#pad digit image with 0
def genPadded(thresh, x, y, w, h):
    # Cropping out the digit from the image corresponding to the current contours in the for loop
    digit = thresh[y:y+h, x:x+w]
    
    # Resizing that digit to (18, 18)
    digitW, digitH = digit.shape
    if digitW > digitH:
        padW = (digitW - digitH) // 2
        padded_digit = np.pad(digit, ((0, 0),(padW, padW)), "constant", constant_values=0)
        #print(0)
    else:
        padW = (digitH - digitW) // 2
        #print(padW)
        padded_digit = np.pad(digit, ((padW ,padW),(0,0)), "constant", constant_values=0)
        
    resized_digit = cv2.resize(padded_digit, (18,18))
    
    # Padding the digit with 5 pixels of black color (zeros) in each side to finally produce the image of (28, 28)
    padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
    return padded_digit

#input a thresholded img and contours of each images
#output megred contours and corresponding img crops
def mergeBeforePrediction(img_thresh, rects): 
    rectsUsed = []
    for cnt in rects:
        rectsUsed.append(False)
        
    # Sort bounding rects by x coordinate
    def getXFromRect(item):
        return item[0]
    
    predIdx = np.array(rects).argsort(axis = 0)[:,0]
    rects.sort(key = getXFromRect)
    
    # Array of accepted rects
    acceptedRects = []
    digit_crops = []
    
    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if (rectsUsed[supIdx] == False):
            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[0] + supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[1] + supVal[3]
    
            # This bounding rect is used
            rectsUsed[supIdx] = True
    
            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):
    
                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[0] + subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[1] + subVal[3]
                #checkoverlap
                overlapyMin = max(curryMin, candyMin)
                overlapyMax = min(curryMax, candyMax)
                overlapxMin = max(currxMin, candxMin)
                overlapxMax = min(currxMax, candxMax)
                #check overlap part
                if (overlapyMax - overlapyMin) / min(subVal[3], supVal[3]) < 0.8 or (overlapxMax - 
                            overlapxMin) / min(subVal[2], supVal[2]) < 0.8:
                    continue
                else:
                    currxMax = max(candxMax, currxMax)
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)
                    # Merge candidate (bounding rect) is used
                    rectsUsed[subIdx] = True
                    break
            x = currxMin
            y = curryMin
            w = currxMax - currxMin
            h = curryMax - curryMin
            #cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 0, 255), thickness=2)
            padded_digit = genPadded(img_thresh, x, y, w, h)
            # Adding the preprocessed digit to the list of preprocessed digits
            digit_crops.append(padded_digit)    
            acceptedRects.append([x, y, w, h])
    return acceptedRects, digit_crops

#*****************************************
#merge two digits to generate a number with two digits, like (1, 2) => 12
#merge is based on distance between contours and merged results.
#if two contours are close to each other and the merged result is a number < 25.
#then you can keep the merged result.
#input (a list of contours of each digits, a list of predictions of each digit)
#output: merged contours and corresponding predictions
def mergeDigits(rects, prediction):
    # Just initialize bounding rects and set all bools to false
    rectsUsed = []
    for cnt in rects:
        rectsUsed.append(False)
    
    # Sort bounding rects by x coordinate
    def getXFromRect(item):
        return item[0]
    
    predIdx = np.array(rects).argsort(axis = 0)[:,0]
    rects.sort(key = getXFromRect)
    
    # Array of accepted rects
    acceptedRects = []
    mergedPrediction = []
    
    # Iterate all initial bounding rects
    for supIdx, supVal in enumerate(rects):
        if rectsUsed[supIdx]:
            continue
        mergedDigit = prediction[predIdx[supIdx]]
        # Initialize current rect
        currxMin = supVal[0]
        currxMax = supVal[0] + supVal[2]
        curryMin = supVal[1]
        curryMax = supVal[1] + supVal[3]

        # This bounding rect is used
        rectsUsed[supIdx] = True

        # Iterate all initial bounding rects
        # starting from the next
        for subIdx, subVal in enumerate(rects[(supIdx+1):], start = (supIdx+1)):
            if rectsUsed[subIdx]:
                continue
            # Initialize merge candidate
            candxMin = subVal[0]
            candxMax = subVal[0] + subVal[2]
            candyMin = subVal[1]
            candyMax = subVal[1] + subVal[3]

            # Check if x distance between current rect
            # and merge candidate is small enough and if merged result is 
            #larger than 25, it should not be merged
            #if digit is 11, they can be seperated farer than others
            dThr = max(supVal[3], subVal[3])
            xThr = dThr / 2
            yThr = dThr / 3
            if prediction[predIdx[supIdx]] == 1:
                xThr = dThr
            
            #the smaller digit should be within the height of the large one, if these two can be 
            #merged
            if (np.abs(candxMin - currxMax) <=  xThr) and ((np.abs(candyMax-curryMax) 
                    <= yThr) or (np.abs(candyMin-curryMin)
                    <= yThr)) and prediction[predIdx[supIdx]] * 10 + prediction[predIdx[subIdx]] < 25:
                overlapMin = max(curryMin, candyMin)
                overlapMax = min(curryMax, candyMax)
                if (overlapMax - overlapMin) / min(subVal[3], supVal[3]) < 0.85:
                    if ((overlapMax - overlapMin) / min(subVal[3], supVal[3]) < 0.7) or (candxMin - 
                                                                                currxMax >=  5):
                        continue
                mergedDigit = prediction[predIdx[supIdx]] * 10  + prediction[predIdx[subIdx]]
                currxMax = max(candxMax, currxMax)
                curryMin = min(curryMin, candyMin)
                curryMax = max(curryMax, candyMax)

                # Merge candidate (bounding rect) is used
                rectsUsed[subIdx] = True
                break
    
            # No more merge candidates possible, accept current rect
            #if currxMax-currxMin <= dThr/4:
            #    continue

        mergedPrediction.append(mergedDigit)
        acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
    return acceptedRects, mergedPrediction

def getDigitContours(grey):
    #threshold for merging two digits is based on image with height of 600
    #need rescale threshold
    imgH, imgW = grey.shape
    ratio = 600 / imgH
    xThr = 18 / ratio

    #grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(grey.copy(), 180, 255, cv2.THRESH_BINARY_INV)  
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if h < xThr:
            continue
        if w/h > 1.4:
            w1 = int(w / 4)
            #cv2.rectangle(image, (x,y), (x+w1, y+h), color=(0, 255, 0), thickness=2)
            cnts.append((x,y,w1,h))
            
            x1 = x + w1
            w2 = 3 * int(w / 4)
            #cv2.rectangle(image, (x1,y), (x1+w2, y+h), color=(0, 255, 0), thickness=2)
            cnts.append((x1,y,w2,h))
        else:  
        
            #cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            cnts.append((x,y,w,h))
    return thresh, cnts


