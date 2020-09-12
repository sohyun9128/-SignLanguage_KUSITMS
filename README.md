# SignLanguage_KUSITMS
## 수화 번역 프로그램  

### 1. handColor.ipynb (손 색 검출)  
 openCV를 통해서 초록 네모 안에 있는 손의 색상 정보를 추출한다.  
 - 손을 초록 네모들 위에 꽉 차게 위치시킨다.  
 - 키보드 'c'를 누르면 손의 색상에 해당하는 영역만 흰 색으로 처리된 창이 나온다.  
 - 원하는 정확도가 나올 때 까지 'c'를 눌러 변경 가능하다.  
 - 's'를 눌러 손의 색상 정보를 저장하고 종료한다. <br><br>

### 2. dataSet.ipynb (데이터셋 생성)  
 저장할 수어 번호, 설명, 이미지를 저장한다  
 - 저장할 수어의 번호(인덱스)와 설명을 저장한다.  
 - 흑백 창이 뜬 후 저장할 손 모양을 창 안에 위치시킨다.  
 - 키보드 'c'를 누르면 total_pics만큼의 이미지가 순차적으로 저장된다. <br><br>
 
### 3. train,test,validation Image.ipynb (이미지를 용도에 따라 분류)  
 전체 이미지의 5/6를 test 이미지로, 1/12를 test 이미지로, 1/12를 validation 이미지로 분류한다. <br><br>

### 4. CNN.ipynb (모델구축 및 학습)  
 keras를 이용하여 CNN모델을 구축하고 학습시킨다.  
 ``` python
   model = Sequential()
   model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
   model.add(Conv2D(32, (3,3), activation='relu'))
   model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
   model.add(Conv2D(64, (5,5), activation='relu'))
   model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
   model.add(Flatten())
   model.add(Dense(128, activation='relu'))
   model.add(Dropout(0.2))
   model.add(Dense(get_num_of_classes()+1, activation='softmax'))
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 ```  
 <br><br>
 ### 5. predict.ipynb (예측 전체)  
 번역을 원하는 손 모양을 해당 위치에 인식시키고 키보드 'e'를 누르면 결과가 저장  
 종료를 원하면 키보드 'f'  
