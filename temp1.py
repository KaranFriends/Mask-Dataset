import os
import shutil
list = []
p = '/media/ayush/Elements/Mask_Recognition/Mask-Dataset/mergedfolder'
i=0
for subdir, dirs, files in os.walk('/media/ayush/Elements/Mask_Recognition/Mask-Dataset/RWMFD_part_2_pro'):
    for path in files:
        (imageClass, fName) = (os.path.basename(subdir), path)
        (imageName, ext) = os.path.splitext(fName)
        list.append(os.path.join(subdir, fName))
        shutil.copy(list[-1],p+'/'+str(i)+ext)
        i+=1
print(len(list))
'''
'''
import cv2
img = cv2.imread('RWMFD_part_2_pro/00000/test.jpg')
cv2.rectangle(img, (145,26),(235,116), (0, 255, 0), 2)
#cv2.rectangle(img, (171,587),(439,854), (0, 255, 0), 2)
while True:
    cv2.imshow('',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
