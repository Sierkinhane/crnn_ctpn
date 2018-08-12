import os
from PIL import Image
import numpy as np

def RmWm(img):

    # height = img.size[0]
    # width = img.size[1]
    print('222')
    if(img.mode == "RGBA"):
        r, g, b, a = img.split()
        #取水印
        vector = np.asarray(r)
        vector.flags.writeable = True               #赋值，给予写的权限
        equal_to_ten_or_five = (vector < 15)
        vector[equal_to_ten_or_five] = 0

        #在a图层中去除水印
        vector_a = np.asarray(a)
        vector_a.flags.writeable = True
        vector_a = vector_a - vector

        #将噪
        equal_r = (vector_a < 50)
        vector_a[equal_r] = 0

        #黑底白字转换
        vector_a = 255-vector_a
        a1 = Image.fromarray(vector_a)
        return a1

    elif img.mode =="RGB":
        r,g,b = img.split()


        arr = np.array(r)

        hist = cv2.calcHist([arr],[0],None,[256],[0.0,255.0])
        minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(hist)
        threshold = max(maxLoc)                             #背景的像素值
        if(threshold >240):
            vector = np.asarray(r)
            vector.flags.writeable = True
            equal_to_ten_or_five = (vector > 220)
            vector[equal_to_ten_or_five] = 255
            out2 = Image.fromarray(vector)
            print('111112')
        else:
            # print(threshold)
            vector = np.asarray(r)
            vector.flags.writeable = True

            vector_r = np.asarray(r)
            vector_r.flags.writeable = True
            length,width = vector_r.shape

            degree = 0
            list1 = []
            list2 = []
            for i in range(1,4):
                list1.append(length//4*i)           #500
                list2.append(width//4*i)            #810

            # print(list1)
            # print(list2)
            # 切分为16个图（并取水印）
            # agg = np.array(g)
            m1 = vector_r[0:int(list1[0]),0:int(list2[0])]
            hist1 = cv2.calcHist([m1], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark1 = max(maxLoc1)  # 背景的像素值
            print(mark1)
            equal_to_ten_or_five3 = (m1 > mark1)
            m1[equal_to_ten_or_five3] = mark1-5
            # equal_to_ten_or_five3 = (m1 < 50) & (m1 > 5)
            # m1[equal_to_ten_or_five3] = m1[equal_to_ten_or_five3] - degree



            m2 = vector_r[0:int(list1[0]), int(list2[0]):int(list2[1]+1)]
            hist1 = cv2.calcHist([m2], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark2 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m2 > mark2)
            m2[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m2 < 50) & (m2 > 5)
            m2[equal_to_ten_or_five3] = m2[equal_to_ten_or_five3] - degree

            m3 = vector_r[0:int(list1[0]), int(list2[1]):int(list2[2] + 1)]
            hist1 = cv2.calcHist([m3], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m3 > mark3)
            m3[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m3 < 50) & (m3 > 5)
            m3[equal_to_ten_or_five3] = m3[equal_to_ten_or_five3] - degree


            m4 = vector_r[0:int(list1[0]), int(list2[2]):r.size[0]+1]
            hist1 = cv2.calcHist([m4], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m4 > mark3)
            m4[equal_to_ten_or_five3] = mark3
            equal_to_ten_or_five3 = (m4 < 50) & (m4 > 5)
            m4[equal_to_ten_or_five3] = m4[equal_to_ten_or_five3] - degree

            m5 = vector_r[int(list1[0]):int(list1[1])+1, 0:int(list2[0])]
            hist1 = cv2.calcHist([m5], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark1 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m5 > mark1)
            m5[equal_to_ten_or_five3] = mark1 - 5
            equal_to_ten_or_five3 = (m5 < 50) & (m5 > 5)
            m5[equal_to_ten_or_five3] = m5[equal_to_ten_or_five3] - degree

            m6 = vector_r[int(list1[0]):int(list1[1])+1, int(list2[0]):int(list2[1] + 1)]
            hist1 = cv2.calcHist([m6], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark2 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m6 > mark2)
            m6[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m6 < 50) & (m6 > 5)
            m6[equal_to_ten_or_five3] = m6[equal_to_ten_or_five3] - degree

            m7 = vector_r[int(list1[0]):int(list1[1])+1, int(list2[1]):int(list2[2] + 1)]
            hist1 = cv2.calcHist([m7], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m7 > mark3)
            m7[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m7 < 50) & (m7 > 5)
            m7[equal_to_ten_or_five3] = m7[equal_to_ten_or_five3] - degree

            m8 = vector_r[int(list1[0]):int(list1[1])+1, int(list2[2]):r.size[0] + 1]
            hist1 = cv2.calcHist([m8], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m8 > mark3)
            m8[equal_to_ten_or_five3] = mark3
            equal_to_ten_or_five3 = (m8 < 50) & (m8 > 5)
            m8[equal_to_ten_or_five3] = m8[equal_to_ten_or_five3] - degree

            m9 = vector_r[int(list1[1]):int(list1[2]) + 1, 0:int(list2[0])]
            hist1 = cv2.calcHist([m9], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark1 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m9 > mark1)
            m9[equal_to_ten_or_five3] = mark1 - 5
            equal_to_ten_or_five3 = (m9 < 50) & (m5 > 5)
            m9[equal_to_ten_or_five3] = m9[equal_to_ten_or_five3] - degree

            m10 = vector_r[int(list1[1]):int(list1[2]) + 1, int(list2[0]):int(list2[1] + 1)]
            hist1 = cv2.calcHist([m10], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark2 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m10 > mark2)
            m10[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m10 < 50) & (m10 > 5)
            m10[equal_to_ten_or_five3] = m10[equal_to_ten_or_five3] - degree

            m11 = vector_r[int(list1[1]):int(list1[2]) + 1, int(list2[1]):int(list2[2] + 1)]
            hist1 = cv2.calcHist([m11], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m11 > mark3)
            m11[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m11 < 50) & (m11 > 5)
            m11[equal_to_ten_or_five3] = m11[equal_to_ten_or_five3] - degree

            m12 = vector_r[int(list1[1]):int(list1[2]) + 1, int(list2[2]):r.size[0] + 1]
            hist1 = cv2.calcHist([m12], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m12 > mark3)
            m12[equal_to_ten_or_five3] = mark3
            equal_to_ten_or_five3 = (m12 < 50) & (m12 > 5)
            m12[equal_to_ten_or_five3] = m12[equal_to_ten_or_five3] - degree

            m13 = vector_r[int(list1[1]):r.size[1] + 1, 0:int(list2[0])]
            hist1 = cv2.calcHist([m13], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark1 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m13 > mark1)
            m13[equal_to_ten_or_five3] = mark1 - 5
            equal_to_ten_or_five3 = (m13 < 50) & (m13 > 5)
            m13[equal_to_ten_or_five3] = m13[equal_to_ten_or_five3] - degree

            m14 = vector_r[int(list1[1]):r.size[1] + 1, int(list2[0]):int(list2[1] + 1)]
            hist1 = cv2.calcHist([m14], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark2 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m14 > mark2)
            m14[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m14 < 50) & (m14 > 5)
            m14[equal_to_ten_or_five3] = m14[equal_to_ten_or_five3] - degree

            m15 = vector_r[int(list1[1]):r.size[1] + 1, int(list2[1]):int(list2[2] + 1)]
            hist1 = cv2.calcHist([m15], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m15 > mark3)
            m15[equal_to_ten_or_five3] = mark2
            equal_to_ten_or_five3 = (m15 < 50) & (m15 > 5)
            m15[equal_to_ten_or_five3] = m15[equal_to_ten_or_five3] - degree

            m16 = vector_r[int(list1[1]):r.size[1] + 1, int(list2[2]):r.size[0] + 1]
            hist1 = cv2.calcHist([m16], [0], None, [256], [0.0, 255.0])
            minVal1, maxVal1, minLoc1, maxLoc1 = cv2.minMaxLoc(hist1)
            mark3 = max(maxLoc1)  # 背景的像素值
            equal_to_ten_or_five3 = (m16 > mark3)
            m16[equal_to_ten_or_five3] = mark3
            equal_to_ten_or_five3 = (m16 < 50) & (m16 > 5)
            m16[equal_to_ten_or_five3] = m16[equal_to_ten_or_five3] - degree

            vector[0:int(list1[0]), 0:int(list2[0])] = m1
            vector[0:int(list1[0]), int(list2[0]):int(list2[1] + 1)] = m2
            vector[0:int(list1[0]), int(list2[1]):int(list2[2] + 1)] = m3
            vector[0:int(list1[0]), int(list2[2]):r.size[0] + 1] = m4

            vector[int(list1[0]):int(list1[1]) + 1, 0:int(list2[0])] = m5
            vector[int(list1[0]):int(list1[1]) + 1, int(list2[0]):int(list2[1] + 1)] = m6
            vector[int(list1[0]):int(list1[1]) + 1, int(list2[1]):int(list2[2] + 1)] = m7
            vector[int(list1[0]):int(list1[1]) + 1, int(list2[2]):r.size[0] + 1] = m8

            vector[int(list1[1]):int(list1[2]) + 1, 0:int(list2[0])] = m9
            vector[int(list1[1]):int(list1[2]) + 1, int(list2[0]):int(list2[1] + 1)] = m10
            vector[int(list1[1]):int(list1[2]) + 1, int(list2[1]):int(list2[2] + 1)] = m11
            vector[int(list1[1]):int(list1[2]) + 1, int(list2[2]):r.size[0] + 1] = m12

            vector[int(list1[1]):r.size[1] + 1, 0:int(list2[0])] = m13
            vector[int(list1[1]):r.size[1] + 1, int(list2[0]):int(list2[1] + 1)] = m14
            vector[int(list1[1]):r.size[1] + 1, int(list2[1]):int(list2[2] + 1)] = m15
            vector[int(list1[1]):r.size[1] + 1, int(list2[2]):r.size[0] + 1] = m16

            out2 = Image.fromarray(vector)
    else:
        r = img
        vector = np.asarray(r)
        vector.flags.writeable = True
        equal_to_ten_or_five3 = (vector>200)
        vector[equal_to_ten_or_five3] = 255
        out2 =Image.fromarray(vector)
    return out2

        
if __name__ == '__main__':

	path = './normal/'
	save_path = './test_vector_version/'

    pictures_name = os.listdir(path)
    for name in pictures_name:
    	srcc =Image.open(path+name)
    	out = RmWm(srcc)
    	#cv2.imshow("image",out)
    	out.save(save_path+name)
    	print(name)
    	#cv2.waitKey(0)