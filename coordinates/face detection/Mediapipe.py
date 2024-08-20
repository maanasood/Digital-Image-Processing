import cv2
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")

image = cv2.imread('/Users/maanassood/study/Projects/Digital Image Processing/img.jpg')

dimensions = image.shape
height = dimensions[0]
width = dimensions[1]
print(f"Height: {height}, Width: {width}")

image = cv2.resize(image, (600, 600), interpolation=cv2.INTER_NEAREST)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    annotated_image = image.copy()
    if results.detections:
        print(f'Number of people detected: {len(results.detections)}')
        print('Description of people:')
        for i, detection in enumerate(results.detections):
            print('Person', i)
            print('Confidence score', detection.score)
            box = detection.location_data.relative_bounding_box

            x_start = int(box.xmin * image.shape[1])
            y_start = int(box.ymin * image.shape[0])
            x_end = int((box.xmin + box.width) * image.shape[1])
            y_end = int((box.ymin + box.height) * image.shape[0])

            annotated_image = cv2.rectangle(annotated_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 5)

        cv2.imshow("Final Image", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No people detected.")
