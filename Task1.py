import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
UBIT = 'sahilsuh'
np.random.seed(sum([ord(c) for
c in UBIT]))
img1 = cv2.imread('mountain1.jpg',0) # queryImage
img2 = cv2.imread('mountain2.jpg',0) # trainImage

'''Reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html'''

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#Task 1.1
image_A = cv2.drawKeypoints(img1,kp1, None)
cv2.imwrite('task1_sift1.jpg',image_A)

image_B = cv2.drawKeypoints(img2,kp2, None)
cv2.imwrite('task1_sift2.jpg',image_B)


# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imwrite("task1_matches_knn.jpg",img3)
plt.imshow(img3),plt.show()

src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

'''Reference: https://docs.opencv.org/3.3.1/d1/de0/tutorial_py_feature_homography.html'''

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = mask, # draw only inliers
                   flags = 2)

matchMask = mask.ravel().tolist()
#matchesMask10 stores the random 10 inliers
matchMask10 = np.zeros(len(matchMask))
counter = 1
while(counter <= 10):
    a = random.randint(0, len(matchMask)-1)
    if(matchMask[a] == 1):
        matchMask10[a] = 1
        counter += 1
        
draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchMask10, # draw only inliers
                   flags = 2)

print("Number of selected inliers: ", np.count_nonzero(matchMask10 == 1))

img4 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imwrite("task1_matches.jpg",img4)
plt.imshow(img4),plt.show()

# Padding the original image to preserve the edge pixels
'''Reference: https://stackoverflow.com/questions/36255654/how-to-add-border-around-an-image-in-opencv-python'''
bordersize = 7
border=cv2.copyMakeBorder(img1, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT )

#Find the corners after the transform has been applied
height, width = border.shape[:2]
corners = np.array([
  [0, 0],
  [0, height - 1],
  [width - 1, height - 1],
  [width - 1, 0]
])

corners = cv2.perspectiveTransform(np.float32([corners]), H)

# Find the bounding rectangle
bx, by, bwidth , bheight = cv2.boundingRect(corners)
bx = -1*bx 
by = -1*by

# Compute the translation homography that will move (bx, by) to (0, 0)
translate = np.array([
  [ 1, 0, bx ],
  [ 0, 1, by],
  [ 0, 0,  1 ]
])
'''This new homography matrix Tr_H preserves the edge pixels, we did this because our original homography 
matrix was cutting off some pixels from the left hand side of the original image. In order 
to have all the pixels intact, we actually translated the original image so that the 
upper leftmost co-ordinate of the translated image comes to (0,0). We then performed 
warp Perspective, 
Reference: http://answers.opencv.org/question/144252/perspective-transform-without-crop/'''
tr_H = translate.dot(H)

#Calculating pano image
warped = cv2.warpPerspective(img1, tr_H, (img1.shape[1] + img1.shape[1] , max(bheight, img1.shape[0])))
warped[bheight-img2.shape[0]:, img2.shape[1]:] = img2
cv2.imwrite("task1_pano.jpg",warped)
plt.imshow(warped),plt.show()