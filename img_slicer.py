#!/usr/bin/env python
# coding: utf-8

# In[4]:


from image_slicer import slice
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

wr, hr = 40, 10
imageArray = [[0 for x in range(wr)] for y in range(hr)]
cleanArray = [[0 for x in range(wr)] for y in range(hr)]
w=0
h=0
counter=0
key=[0, 0, 0]
file = "../image/out.png"
img = Image.open(file)
thresh = 200
fn = lambda x : 255 if x > thresh else 0
r = img.convert('L').point(fn, mode='1')
r.save(file)
imgRead = cv2.imread(file)
ar = np.array(imgRead)
for r in range(0,imgRead.shape[0],10):
    # print("petla pierwsza wykonalem sie h=",h," raz")
    for c in range(0,imgRead.shape[1],10):
        # print("petla druga wykonalem sie w=",w," raz")
        cv2.imwrite(f"../imageSplitted/img{r}_{c}.png",imgRead[r:r+10, c:c+10,:])
        image = Image.open(f"../imageSplitted/img{r}_{c}.png")
        imageArray[h][w]=np.array(image)
        if np.any(imageArray[h][w] == key) != True:
            cleanArray[h][w]=0
        else:
            counter+=1
            cleanArray[h][w]=1
        try: 
            os.remove(f"../imageSplitted/img{r}_{c}.png")
        except: pass   
        w+=1         
    h+=1
    w=0
# for c in cleanArray:
#     print(c)

arr=np.sum(cleanArray,axis=0)
# print()
# print (arr)
sum=0
List= []
ListOfArrays= []
i=0
j=0
letterLength=0
for e in range (40):
    if arr[e] !=0:
        arr[e]=1
    
# print (arr)
for ee in range (40):
    if arr[ee] != 0:
        if(arr[ee]*arr[ee+1]==0):
            letterLength+=1
            # print("last",ee)
            List.append(ee)
        else:
            letterLength+=1
        # print(letterLength)
    else:
        # print(0)
        if(ee<39):
            if(arr[ee]+arr[ee+1]==1):
                # print("first",ee)
                List.append(ee+1)
                letterLength=0

letters=(len(List))/2
# print(letters)
# print(List)   

# for rows in range (10):  # outer loop  
#         for columns in range (40):  # inner loop
#             print(cleanArray[rows][columns], end = " ") # print the elements 
#         print() 
wrr, hrr = 40, 10      
for xx in range (int(letters)):
    sthArray = [[0 for y in range(List[1+(2*xx)]-List[0+(2*xx)]+1)] for x in range(hr)]
    # print("nowa iteracja")
    for rows in range (10):  # outer loop  
        for columns in range (40):  # inner loop
            if columns>= List[0+(2*xx)] and columns  <= List[1+(2*xx)]:
                sthArray[rows][columns-List[0+(2*xx)]]=cleanArray[rows][columns]
                # print("kolumna: ",columns)
                # print("wiersz: ",rows)
    # for c in sthArray:
    #     print(c)
    ListOfArrays.append(sthArray)


### dzielenie liter do zdjec 28x28 ###
black = [0, 0, 0]
white = [255, 255, 255]
width = 28
height= 28

for x in range (len(ListOfArrays)):
    arrayOfLetters=np.array(ListOfArrays[x])
    rows = len(arrayOfLetters)
    columns = len(arrayOfLetters[0])
    # print (rows, columns)
    np_imgTmp = [[white for x in range(columns)] for y in range(rows)]
# arrayOfLettersRGB=np.array([0, 0, 0])
# print(arrayOfLetters[0][2])
    imga = Image.fromarray(arrayOfLetters, 'RGB')

# true_mask = np.all(arrayOfLetters == target_color, axis=2)
# false_mask = np.all(img != target_color, axis=2)

    for i in range (rows):
        for j in range (columns):
            if arrayOfLetters[i][j].any() == 1:
                # counterr+=1
                np_imgTmp[i][j]=black
# print(np_imgTmp)
# imss=Image.fromarray(np_imgTmp)
# imss.show()
    npp_imgTmp= np.array(np_imgTmp)
# np.iinfo(npp_imgTmp.dtype)
    img = npp_imgTmp.astype(np.uint8)
    img=Image.fromarray(img)
    img = img.resize((width, height), Image.ANTIALIAS)
    img.save(f'../letters/letter{x}.png')
    # img.show() pokaz obrazek


# In[ ]:




