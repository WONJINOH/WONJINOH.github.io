```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import os

```


```python
my_image_path = os.getenv('HOME')+ '/aiffel/camera_sticker/images/ohwonjin.jpg'
img_bgr = cv2.imread(my_image_path)   
img_show = img_bgr.copy()      
plt.imshow(img_bgr)
plt.show()

#matplotlib,dlib에서는 모두 RGB 채널을 사용하는 반면
#opencv에서는 예외적으로 BGR(ble, green, red)의 채널을 사용하기에 푸르게 보임
#따라서 보정처리를 필요로함.
```


    
![png](output_1_0.png)
    



```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR 이미지를 RGB 이미지로 변경
plt.imshow(img_rgb)
plt.show()
```


    
![png](output_2_0.png)
    



```python
# 디턱테 선언
detector_hog = dlib.get_frontal_face_detector() 
```


```python
# 디텍터를 이용해서 얼굴의 BOUNDING BOX를 추출
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)  
```


```python
print(dlib_rects)   # 찾은 얼굴영역 좌표

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```

    rectangles[[(348, 400) (810, 862)]]



    
![png](output_5_1.png)
    



```python
#얼굴 랜드 마크
model_path = 'models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)    
# 저장된 landmark 모델 불러오기
```


```python
list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))

```

    68



```python
for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1) # yellow

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()
```


    
![png](output_8_0.png)
    



```python
# 3. 스티커 적용
for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[33]) # nose center index : 30
    x = landmark[33][0]
    y = landmark[33][1]
    w = dlib_rect.width()
    h = dlib_rect.width()
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))
```

    (599, 672)
    (x,y) : (599,672)
    (w,h) : (463,463)



```python
sticker_path = 'images/cat.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w,h))
print (img_sticker.shape)
```

    (463, 463, 3)



```python
refined_x = x - w // 2  # left
refined_y = y - h // 2  # top
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```

    (x,y) : (368,441)



```python
# img_sticker = img_sticker[-refined_y:]  # -y 크기 만큼 스티커를 crop
# refined_y = 0  #  y 좌표는 원본 이미지의 경계 값으로 수정
# print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```


```python
if refined_x < 0:
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
if refined_y < 0:
    img_sticker = img_sticker[-refined_y:, :]
    refined_y = 0
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))
```

    (x,y) : (368,441)



```python
sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,img_sticker, sticker_area).astype(np.uint8)
print('하이')
```

    하이



```python
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()
```


    
![png](output_15_0.png)
    



```python
sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] = \
    np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()
```


    
![png](output_16_0.png)
    


<회고>
1. 처음 다른 사진을 적용하여 왕관을 적용했을 때 머리의 끝이 사진 상단에 위치하여 왕관의 위치를 조절하였음에도 왕관이 짤리는 모습이 관찰됨.
이에, 사진을 다른 사진으로 교환하였음.
2. 수염이 머리 상단에 왕관처럼 걸쳐져 안내려오는 문제가 발생함. refined_y = y - h // 2로 변경해도 수정되지 않았으나.17번에서 문제를 발견하고 수정함.

