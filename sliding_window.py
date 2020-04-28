import cv2
import numpy as np
import numpy_indexed as npi
from scipy.cluster.vq import vq, kmeans
from numpy import median
import time as t
import pandas as pd
#from collections import namedtuple

# load the image and define the window width and height


def TrackingClr(lst1,lstName,filename,wrkstation,WrkstnNm,roiframe,frameNoPS):
    
    def area1(ax,ay,ax1,ay1,bx,by,bx1,by1):  # returns None if rectangles don't intersect
        dx = min(max(ax,ax1), max(bx,bx1)) - max(min(ax,ax1), min(bx,bx1))
        dy = min(max(ay,ay1), max(by,by1)) - max(min(ay,ay1), min(by,by1))
        area=dx*dy
        if area>0:
            return True
        else:
            return False
    def do_cluster(hsv_image, K, channels):
        # gets height, width and the number of channes from the image shape
        h,w,c = hsv_image.shape
        # prepares data for clustering by reshaping the image matrix into a (h*w) x c matrix of pixels
        cluster_data = hsv_image.reshape( (h*w,c) )
        # performs clustering
        
        codebook, distortion = kmeans(np.array(cluster_data[:,0:channels], 
                                               dtype=np.float), K)
        
        codebook = list(codebook)
        if len(codebook) != K:
            for i in range(0,abs(K-len(codebook))):
                codebook.append(codebook[0])
        return codebook
    
    roiLst = []
    hsvroilst = []
    print(filename)
    cap = cv2.VideoCapture(filename)
    #here we are calculating fps of video 
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ok, frme = cap.read()
    frame1 = roiframe
    rows,cols = frame1.shape[:2]
    out = cv2.VideoWriter('Workshop8.avi',cv2.VideoWriter_fourcc('M','J','P','G'),
                      10, (cols,rows))
    #here we are cropping roi in first frame 
    for z in lst1:
        roiLst.append(frame1[int(z[1]):int(z[3]),int(z[0]):int(z[2])])
        img = frame1[int(z[1]):int(z[3]),int(z[0]):int(z[2])]
        imghsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hsvroilst.append(do_cluster(imghsv, 5, 3))
    kernel = np.ones((3, 3), np.uint16)
    #this function is used for intersection of two arrays
    
    def array_row_intersection(a,b):
       tmp=np.prod(np.swapaxes(a[:,:,None],1,2)==b,axis=2)
       return a[np.sum(np.cumsum(tmp,axis=0)*tmp==1,axis=1).astype(bool)]
    # this function is use to cluster the hsv image pixels
    length_wrk=len(wrkstation)
    WorkList =  [[] for _ in range(length_wrk)]
    # this variable is use for count frame
    frm = 1
    UpdtIndx = [[0,0] for i in range(len(roiLst))]
    UpdtNme = [ " " for i in range(len(roiLst))]
    
    df=pd.DataFrame(columns=[str(i) for i in WrkstnNm],index=[str(j) for j in lstName])
    df[:]=0
    #print(dic)
    while True:
        t0 = t.time()
        ret, frame = cap.read()
        if frame is None:
            break
        #Below line we are doing background subtraction
        
        fgmask2 = cv2.absdiff(frme, frame)
        
        fgmask2 = cv2.cvtColor(fgmask2, cv2.COLOR_BGR2GRAY)
        fgmask2 = cv2.medianBlur(fgmask2, 1)
        fgmask2 = cv2.GaussianBlur(fgmask2, (1, 1), 0)
        ret3, fgmask2 = cv2.threshold(
            fgmask2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dilation = cv2.dilate(fgmask2, kernel, iterations=7)
        # In below line we get contours from dilated images
        im2,contours,hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                      cv2.CHAIN_APPROX_SIMPLE)
        filteredContours = [] 
        '''countPerson=[]
        countPerson1=[]'''
        #In below for loop we are ignoring those loop, which area is less than 500
        for j in contours:
            area=cv2.contourArea(j)
            perimeter = cv2.arcLength(j,True)
            if area>500 and perimeter>500 :
            		filteredContours.append(j)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []
        cntr = 1
        binx = 0
        biny = 0
        print("")
        for contour, hier in zip(filteredContours, hierarchy):
            #print("-*-*-*-*-*-*-*-*-*Counters" + str(cntr) + "-*-*-*-*-*-*-*-**-")
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(frame,(x,y),((x+w),(y+h)), (255,0,0), 2)
            if w > 5 and h > 5:
                DenChekx = [[] for i in range(len(roiLst))]
                DenCheky = [[] for i in range(len(roiLst))]
                DenChek = [0 for i in range(len(roiLst))]
                DenChekRdndncy = [0 for i in range(len(roiLst))]
                """UpdtIndx = [[0,0] for i in range(len(roiLst))]
                UpdtNme = [ " " for i in range(len(roiLst))]"""
                for k in range(len(roiLst)):
                    
                    #if roiLst[k] not in countPerson:
                    img = roiLst[k]
                    #print("*-*-**-*-------Roi-*-*--*-*--*-*-")
                    #imghsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                    roilst = hsvroilst[k]
                    rows, cols = img.shape[:2]
                    rows, cols = int(rows*.6), int(cols*.6)
                    frameTest = frame[y:y+h,x:x+w]
                    binx = x
                    biny = y
                    #print("-*-*-*--*-*-*-*-*-*-*-*Bins-*-*-*-*-*-*-*-*")
                    #here we are broking the contours in bins, which size is 60% of roi size
                    cnt = 0
                    for r in range(0,frameTest.shape[0] - rows, rows):
                        for c in range(0,frameTest.shape[1] - cols, cols):
                            cnt = cnt + 1
                            #DenChek = [0] * len(roiLst)
                            if cnt%4 == 0:
                                image = frameTest[r:r+rows,c:c+cols]
                                imagehsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
                                binlst = do_cluster(imagehsv, 2, 3)
                                
                                #here we are matching roi hsv value with bins hsv value
                                if ((abs(int(binlst[0][0])- int(roilst[0][0])) < 5
                                     and abs(int(binlst[0][1])- int(roilst[0][1])) < 5 
                                     and abs(int(binlst[0][2])-int(roilst[0][2])) < 5) or
                                    (abs(int(binlst[1][0])- int(roilst[0][0])) < 5 
                                     and abs(int(binlst[1][1])- int(roilst[0][1])) < 5 
                                     and abs(binlst[1][2]-roilst[0][2]) < 5)):
                                    DenChekx[k].append(int(binx + c + cols/2))
                                    DenCheky[k].append(int(biny + r + rows/2))
                                if ((abs(int(binlst[0][0])- int(roilst[1][0])) < 5 
                                     and abs(int(binlst[0][1])-int(roilst[1][1])) < 5
                                     and abs(int(binlst[0][2])- int(roilst[1][2])) < 5) or
                                    (abs(int(binlst[1][0])-int(roilst[1][0])) < 5 
                                     and abs(int(binlst[1][1])- int(roilst[1][1])) < 5
                                     and abs(int(binlst[1][2])- int(roilst[1][2])) < 5)):
                                    DenChekx[k].append(int(binx + c + cols/2))
                                    DenCheky[k].append(int(biny + r + rows/2))
                                if ((abs(int(binlst[0][0])- int(roilst[2][0])) < 5 
                                     and abs(int(binlst[0][1])- int(roilst[2][1])) < 5 
                                     and abs(int(binlst[0][2])- int(roilst[2][2])) < 5) or
                                    (abs(int(binlst[1][0])- int(roilst[2][0])) < 5 
                                     and abs(int(binlst[1][1])- int(roilst[2][1])) < 5
                                     and abs(int(binlst[1][2])- int(roilst[2][2])) < 5)):
                                    DenChekx[k].append(int(binx + c + cols/2))
                                    DenCheky[k].append(int(biny + r + rows/2))
    
                                if ((abs(int(binlst[0][0])- int(roilst[3][0])) < 5 
                                     and abs(int(binlst[0][1])- int(roilst[3][1])) < 5
                                     and abs(int(binlst[0][2])- int(roilst[3][2])) < 5) or
                                    (abs(int(binlst[1][0])- int(roilst[3][0])) < 5 
                                     and abs(int(binlst[1][1])- int(roilst[3][1])) < 5
                                     and abs(int(binlst[1][2])- int(roilst[3][2])) < 5)):
                                    DenChekx[k].append(int(binx + c + cols/2))
                                    DenCheky[k].append(int(biny + r + rows/2))
    
                                if ((abs(int(binlst[0][0])- int(roilst[4][0])) < 5 
                                     and abs(int(binlst[0][1])- int(roilst[4][1])) < 5 
                                     and abs(int(binlst[0][2])- int(roilst[4][2])) < 5) or
                                    (abs(int(binlst[1][0])- int(roilst[4][0])) < 5 
                                     and abs(int(binlst[1][1])-int(roilst[4][1])) < 5
                                     and abs(int(binlst[1][2])- int(roilst[4][2])) < 5)):
                                    DenChekx[k].append(int(binx + c + cols/2))
                                    DenCheky[k].append(int(biny + r + rows/2))
                                #In below lines code we are matching pixel value channel vise
                                #and than find intersection of all matched pixel"""
                    
                                idx1= np.isin(image[:,:,0], img[:,:,0])
                                lst1 = np.where(idx1)
                                val1 = np.array([lst1[1]+c+x,lst1[0]+r+y])
                                lst11 = np.transpose(val1)
                                idx2= np.isin(image[:,:,1], img[:,:,1])
                                lst2 = np.where(idx2)
                                val2 = np.array([lst2[1]+c+x,lst2[0]+r+y])
                                lst22 = np.transpose(val2)
                                idx3= np.isin(image[:,:,2], img[:,:,2])
                                lst3 = np.where(idx3)
                                val3 = np.array([lst3[1]+c+x,lst3[0]+r+y])
                                lst33 = np.transpose(val3)
                                lstintr = (array_row_intersection(lst22,lst33))
                                lstrfinal = array_row_intersection(lstintr,lst11)
                                #here we are incrementing the value when bin is match 
                                #to roi's 2/3 pixels values
                                if len(lstrfinal) > int((rows*cols*2)/3):
                                    DenChek[k] = DenChek[k]+1
                                  
                        
                        #here we are checking condition on hsv value if it is match
                        #to the roi hsv of that person than show the in name of that person 
                        #in output frame
                        #countPerson.append(roiLst[k])
                    if len(DenChekx[k]) > 0:
                        #if roiLst[k] not in countPerson1:
                        for j in range(len(wrkstation)):
                            roiOverlap=area1(int(x),int(y),int(x+w),int(y+h),int(wrkstation[j][0]), int(wrkstation[j][1]),int(wrkstation[j][0]+wrkstation[j][2]),
                                            int(wrkstation[j][1]+wrkstation[j][3]))
                            if roiOverlap==True:
                               '''WorkList[WrkstnNm[j]].append(lstName[k])'''
                               WorkList[j].append(str(lstName[k]))
                        
            
                        xbn = median(DenChekx[k])
                        ybn = median(DenCheky[k])
                        cv2.putText(frame,lstName[k],(int(xbn),int(ybn))
                                            ,0,1,(0,0,255),2,cv2.LINE_AA)
                        DenChekRdndncy[k] = DenChekRdndncy[k] +1
                        UpdtIndx[k][0] = int(xbn)
                        UpdtIndx[k][1] = int(ybn)
                        UpdtNme[k] = lstName[k]
                        for nm in lstName:
                            if frm%(3*fps) == 0 and nm == lstName[k]:
                                 '''print("Position of "+lstName[k]+" is :- "+
                                       str((int(xbn),int(ybn))))'''
                            #countPerson1.append(roiLst[k])
                #here we are claculating the max number pixel mathced with contour(In
                #terms of bins image)and show the name of that person which shirt or tshirt
                #roi is matched maximum time 
                MaxRoi = max(DenChek)
                MaxRoiIndx = DenChek.index(MaxRoi)
                if DenChek[MaxRoiIndx] > 0 and DenChekRdndncy[MaxRoiIndx] == 0:
                    cv2.putText(frame,lstName[MaxRoiIndx],(int((w/2) + x),
                                int((h/2) + y)),0,1,(0,0,255),2,cv2.LINE_AA)
                    UpdtIndx[k][0] = int(w/2 + x)
                    UpdtIndx[k][1] = int(h/2 + y)
                    UpdtNme[k] = lstName[MaxRoiIndx]
                    for j in range(len(wrkstation)):
                            roiOverlap=area1(int(x),int(y),int(x+w),int(y+h),int(wrkstation[j][0]), int(wrkstation[j][1]),int(wrkstation[j][0]+wrkstation[j][2]),
                                            int(wrkstation[j][1]+wrkstation[j][3]))
                            if roiOverlap==True:
                                '''print(lstName,lstName[k])
                                Name=str(lstName[k])
                                WorkList.insert(j[z],str(Name))'''
                                WorkList[j].append(str(lstName[MaxRoiIndx]))
                               
                    for nm in lstName:
                            if frm%(3*fps) == 0 and nm == lstName[MaxRoiIndx]:
                                '''print("Position of "+lstName[MaxRoiIndx]+
                            " is :- "+str((int((w/2) + x),int((h/2) + y))))'''
                """else:
                    #if above both condition are failed than here we are updating the 
                    #person name with pervious data
                    try:
                        for m in range(len(roiLst)):
                            cv2.putText(frame,UpdtNme[m],(UpdtIndx[m][0],UpdtIndx[m][1])
                            ,0,1,(0,0,255),2,cv2.LINE_AA)
                        for nm in lstName:
                            if frm%(3*fps) == 0 and nm == UpdtNme[0]:
                                print("Position of "+lstName[m]+" is :- "+
                                      str((UpdtIndx[m][0],UpdtIndx[m][1])))
                    except:
                        pass"""
                
            cntr = cntr+1
        
        if(frm%int(frameNoPS)*10==0):
            print(WorkList)
            for i in range(len(WorkList)):
            
                print("mmmmmmmmmmmm")
                for j in range(len(WorkList[i])):
                    
                    print("ppppppppppp")
                    df.loc[WorkList[i][j]][WrkstnNm[i]]=df.loc[WorkList[i][j]][WrkstnNm[i]]+1 
                '''MaxPeople=max(WorkList[i],key=WorkList[i].count)
                textFile="The person "+MaxPeople+" is working on "+WrkstnNm[i]+"\n"
                
            
                
                print("The person ",MaxPeople," is working on ",WrkstnNm[i])'''
                print(df)
                WorkList[i].clear()
                  
                    
            
        #here we are showing the workstation bounding box and its name on the frame
        for g in range(len(wrkstation)):
            cv2.rectangle(frame, (int(wrkstation[g][0]), int(wrkstation[g][1]))
            , (int(wrkstation[g][2]), int(wrkstation[g][3])), (0,255,255), 2)
            cv2.putText(frame,WrkstnNm[g],(int((int(wrkstation[g][0])+
                            int(wrkstation[g][2]))/2), int((int(wrkstation[g][1])
                            +int(wrkstation[g][3]))/2))
            ,0,1,(0,0,255),2,cv2.LINE_AA)
       # print("lstName",lstName)
        
        cv2.imshow("FinalOutput",frame)
       # print(WorkList)
        out.write(frame)
        frm = frm + 1
        
        t1 = t.time()
        #print("Clusterization took %0.5f seconds" % (t1-t0))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    df=df[:]/10
    df.to_csv("Timesheet.csv", sep=',', encoding='utf-8', index=True)
    
def SelectFrame(filename):
    cap = cv2.VideoCapture(filename)
    while True:
        ok,frame = cap.read()
        cv2.imshow("selectOutput",frame)
        if cv2.waitKey(100) & 0xFF == ord('s'):
            break
    cv2.destroyAllWindows()
    cap.release()
    return frame