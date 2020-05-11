import cv2
import os
from collections import defaultdict

orb=None
flann=None
paintings=[]

#Fill 'paintings' array with all pictures file names
def loadFiles():
    global paintings
    paintings = os.listdir("images")

#Initializes ORB and Flann
def initOrbandFlann():
    global orb,flann
    orb = cv2.ORB_create(700)   #ORB Init
    index_det = dict(algorithm = 6,table_number=10,key_size=20,multi_probe_level=0)
    search = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_det,search) #Flann Init

#Get descriptors for each images
def getDescriptors():
    global flann,paintings
    for filename in paintings:
        img2 =cv2.imread("images/"+filename,0)
        kp2, des2 = orb.detectAndCompute(img2, None)
        flann.add([des2])#Add each image descriptor to flann
    flann.train()
#Get a match for a image
#img -> filename of the painting
def getMatch(img):
    global kp1, des1,flann
    img1 = cv2.imread(img)
    kp1, des1 = orb.detectAndCompute(img1, None)
    matches= flann.match(des1)
    matches_dict = defaultdict(lambda : 0)
    for f in matches:
        matches_dict[f.imgIdx]+=1
    temp = sorted(matches_dict.items(),key=lambda x:x[1],reverse=True)
    index = next(iter(temp))
    file = paintings[list(index)[0]]
    print(file)

def main():
    global flann
    print("Initialating ORB")
    initOrbandFlann()
    print("Loading files")
    loadFiles()
    print("Getting images descriptors")
    getDescriptors()
    print("Training FLANN")
    print("Current number of paintings: "+str(len(flann.getTrainDescriptors())))
    while True:
        image = input("Write image name\n")
        print("Getting match")
        getMatch(image)

main()



