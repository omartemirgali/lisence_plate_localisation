import os
import cv2
import numpy as np
import imutils


def detect_plate(img_path):
    try:
        image = cv2.imread(img_path)
        image = imutils.resize(image, width=500)

        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noiseless_image = cv2.fastNlMeansDenoising(grayscale_image, None, 20, 7, 21) 
        smoothed_image = cv2.GaussianBlur(noiseless_image, (5, 5), 0)
        edged_image = cv2.Canny(smoothed_image, 30, 200)

        contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:30]

        plate_contour = None

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            concat = np.concatenate(contour)
            hulls = cv2.convexHull(concat)
            approx = cv2.approxPolyDP(hulls, 0.018 * perimeter, True)
            if len(approx) == 4: 
                plate_contour = contour
                break
        
        x, y, w, h = cv2.boundingRect(plate_contour) 
        cropped_image = image[y: y + h, x: x + w]
        image_name = img_path.replace('./input/', '').replace('.jpg', '')

        if not (os.path.exists('./output')):
            os.makedirs('./output')
            cv2.imwrite('./output/' + image_name + '.png', cropped_image)
        else: 
            cv2.imwrite('./output/' + image_name + '.png', cropped_image)

    except cv2.error as e:
        if e.err == "!_img.empty()":
            print("Could not load the image " + image_name)
            pass


def main():
    for images in os.listdir('./input'):
        if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg") or images.endswith(".JPG")):
            detect_plate('./input/' + images)
    

if __name__ == "__main__":
    main()