import cv2
import os
import shutil
#ฟังก์ชั่นเปรียบเทียบใบหน้า
def compare_faces(face1, face2):
    diff = cv2.norm(face1, face2, cv2.NORM_L2)
    return diff < 100
#Folderที่เก็บรูปภาพที่จะเอาไปเทียบ
folder_path = "data"
#Folderที่เก็บรูปภาพที่ได้ใบหน้าเหมือนกันแล้ว
output_folder = "out" #ถ้ายังไม่มีFolder out ให้สร้าง out ขึ้นมา
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

reference_face_path = "Pic.jpg" #ประกาศรูปอ้างอิงไว้เปรียบเทียบ
reference_face = cv2.imread(reference_face_path)
#เรียกใช้haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#เปลี่ยนสีรูปอ้างอิงเป็นสีเทา (เพื่อให้Haarcascadeตรวจจับได้ดีขึ้น)
gray_reference_face = cv2.cvtColor(reference_face, cv2.COLOR_BGR2GRAY)
#ตรวจหาใบหน้าอ้างอิง
detected_reference_faces = face_cascade.detectMultiScale(gray_reference_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#ถ้าเจอใบหน้าที่เหมือนกันกับใบหย้าอ้างอิง ให้ทำตามข้างล่าง
if len(detected_reference_faces) > 0:
    largest_reference_face = max(detected_reference_faces, key=lambda x: x[2] * x[3]) #หาใบหน้าที่ใหญ่ที่สุดไว้เปรียบเทียบ    
    x_ref, y_ref, w_ref, h_ref = largest_reference_face #บรรทัดที่26 27 ตีกรอบเพื่อหาใบหน้า
    reference_face_cropped = reference_face[y_ref:y_ref+h_ref, x_ref:x_ref+w_ref]

    for filename in os.listdir(folder_path): #loopตามรูปใน data
        image_path = os.path.join(folder_path, filename) #เลือกเส้นทางของรูป
        image = cv2.imread(image_path) #อ่านรูป

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #ประกาศตัวแปลไว้แปลงรูปเป็นสีเทา    

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) #หาใบหน้าที่ใช้ตัวแปรรูปเป็นรูปอ้างอิง

        is_face_matched = False #is_face_matchedไว้ตรวจว่ารูปเหมือนกันหรือไม่ถ้าไม่จะloopต่อจนกว่าจะเจอรูปที่มีใบหน้าเหมือนกัน  
        for (x, y, w, h) in faces:
            face_cropped = image[y:y+h, x:x+w] #ครอบลูกศรบนใบหน้าที่เจอในรูปที่จะสแกน
            # ปรับขนาดให้มีขนาดเดียวกันกับใบหน้าที่ใช้เปรียบเทียบ
            face_cropped_resized = cv2.resize(face_cropped, (w_ref, h_ref))
            #เทียบใบหน้ากับรูปอ้างอิง
            if compare_faces(face_cropped_resized, reference_face_cropped):
                is_face_matched = True #เปลี่ยนจากFalse เป็น True 
                break #หยุดloop

        if is_face_matched: #ถ้ามีใบหน้าที่เหมือนในรูป
            shutil.copy(image_path, os.path.join(output_folder, filename)) #copyรูปนั้นไปไว้ในout
            print(f"บันทึก: {filename}")
#บรรทัด 49 51 พอcopyจะให้แสดงว่าบันทึก+ชื่อไฟล์นั้น พอเสร็จสิ้นกระบวยการให้แสดง เสร็จสิ้นการแยกรูป
print("เสร็จสิ้นการแยกรูป")
