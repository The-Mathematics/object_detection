import cv2
from ultralytics import YOLO



model = YOLO("yolov8s.pt")

#Create a variable to store the resulting detection by specifying the file path for the image to be analysed by the model and a list of items you want to detect
images = [cv2.imread("car-image.jpeg"), cv2.imread("office image.jpeg"), cv2.imread("people-image.jpeg")]

#for image in images:
results = model.predict(images, conf=0.5, classes=[0, 1, 2, 3, 4, 5, 9, 13, 15, 16, 24, 26, 27, 28, 32, 41, 56, 57, 62, 63, 64, 66, 67,73, 74])

#This runs the model on each image parsed
for i,result in enumerate(results):
    objects_in_model = result.names
    bounding_boxes = result.boxes.xyxy.tolist()
    list_of_classes_detected = result.boxes.cls.tolist()
    confidence = result.boxes.conf.tolist()

#Count for each object to be detected with an initial value of 0
    count_dog = 0
    count_cat = 0
    count_chair = 0
    count_car = 0
    count_bicycle = 0
    count_motorcycle = 0
    count_bus = 0
    count_trafficLight = 0
    count_bench = 0
    count_backpack = 0
    count_handbag = 0
    count_tie = 0
    count_suitcase = 0
    count_sportsball = 0
    count_cup = 0
    count_couch = 0
    count_tv = 0
    count_laptop = 0
    count_mouse = 0
    count_keyboard = 0
    count_cellphone = 0
    count_book = 0
    count_clock = 0
    count_person = 0
    count_airplane = 0

# This checks to see if the objects specified are in the image and then increments the count if present
    for class_detected in (list_of_classes_detected):
        label = objects_in_model[class_detected]
        if label == "chair":
            count_chair += 1
        elif label == "car":
            count_car += 1
        elif label == "person":
            count_person += 1
        elif label == "book":
            count_book += 1
        elif label == "traffic light":
            count_trafficLight += 1
        elif label == "tv":
            count_tv += 1
        elif label == "keyboard":
            count_keyboard += 1
        elif label == "mouse":
            count_mouse += 1
        elif label == "suitcase":
            count_suitcase += 1
        elif label == "tie":
            count_tie += 1
        elif label == "laptop":
            count_laptop += 1
        elif label == "cellphone":
            count_sportsball += 1
        elif label == "cup":
            count_cup += 1
        elif label == "couch":
            count_couch += 1
        elif label == "clock":
            count_clock += 1
        elif label == "bus":
            count_bus += 1




#This creates variables containing the object detected and it's count
        objectCount = "people:" + str(count_person) + " " + "cars:" + str(count_car) + " " + "buses:" + str(count_bus) +" " + "chair:" + str(count_chair)  + " " + "tv:" + str(count_laptop) + " " + "keyboard:" + str(count_keyboard) + " " + "book:" + str(count_book)
#This puts a text displaying the objects we want to detect in each image and their count
    cv2.putText(images[i], str(objectCount), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
#This prints the classes detected for each image
    print(list_of_classes_detected)
#This displays the results foe each image
    result.show()

# print the names of objects detected
    print("list of classes detected:", list_of_classes_detected)
    print("bounding_boxes:", bounding_boxes)
    print("confidence:", confidence)
    print("---")




cv2.waitKey(0)
cv2.destroyAllWindows()