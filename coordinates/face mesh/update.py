import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import json

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

image = cv2.imread('/Users/maanassood/study/Projects/Digital Image Processing/img.jpg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

h, w, _ = image.shape
print(h, w)
result = face_mesh.process(rgb)
print(result)

coordinates_list = []

for facial_landmarks in result.multi_face_landmarks:
    for i in range(0, 468):
        pt = facial_landmarks.landmark[i]
        x = pt.x*w
        y = pt.y*h

        result = "{{x:{}, y:{}}}".format(x, y)
        result_dict = json.loads(result.replace('x', '"x"').replace('y', '"y"'))

        print(result)
        print(result_dict)
        print(type(result_dict))

        coordinates_list.append(result_dict)
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 0), 10)
print(coordinates_list)

def convert_to_json(list):
    json_data = json.dumps(list)

    with open("my_list.json","w") as outfile:
        outfile.write(json_data)
convert_to_json(coordinates_list)

def show(title='Images', image=None, size=10):
    w, h = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size*aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
show('After face mesh', image)
