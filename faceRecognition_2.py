import cv2
import face_recognition
import os   #load thu vien anh
import numpy as np

# Buoc 1 load ảnh từ kho ảnh nhận dạng

#đường dẫn
path = "pic2"   #đường dẫn
images = []
classNames = []
myList = os.listdir(path)   #Kiểm tra toàn bộ tên file ở trong pic2
print(myList)  # ['Donal Trump.jpg', 'elon musk .jpg', 'Joker.jpg', ...]
for i in myList:
    curImg = cv2.imread(f"{path}/{i}")  # pic2/Donal Trump.jpg(đẩy về 1 ma trận điểm ảnh)
    images.append(curImg)
    classNames.append(os.path.splitext(i)[0])
    # splitext sẽ tách path ra thành 2 phần, phần trước đuôi mở rộng và phần mở rộng
    # để lấy tên của bức ảnh
print(len(images))
print(classNames)


# Buoc 2 encoding
def Mahoa(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR được chuyển đổi sang RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnow = Mahoa(images)
print("Ma hoa thanh cong")
print(len(encodeListKnow))

# khởi dộng webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()#(ret  load đúng thì trả về true)
    framS = cv2.resize(frame,(0, 0), None, fx=0.5, fy=0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    # xác định vị trí khuôn mặt trên cam và encode hình ảnh trên cam
    facecurFrame = face_recognition.face_locations(framS)  # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodecurFrame = face_recognition.face_encodings(framS)
#Buoc 3: Kiem tra nhan dang
    for encodeFace, faceLoc in zip(encodecurFrame,
                                   facecurFrame):  # lấy từng khuôn mặt và vị trí khuôn mặt hiện tại theo cặp
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)  # đẩy về index của faceDis nhỏ nhất

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
        else:
            name = "Unknow"

        # print tên lên frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ
