import numpy as np
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, detect_conf = 0.5, track_conf = 0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=detect_conf,
            min_tracking_confidence=track_conf,
        )

    def get_landmarks(self, img):
        # img should be in RGB
        h, w, c = img.shape
        results = self.hands.process(img).multi_hand_landmarks
        if not results:
            return None
        else:
            return np.array([[[int(lm.x*w),int(lm.y*h),int(lm.z*c)] for lm in handLms.landmark] for handLms in results])


    def resize_pad_concat(imgs, handsize=200, size=224):
        images = []
        for image in imgs:
            max_shape = np.max(image.shape[:2])
            scale_percent = handsize / max_shape
            width = round(image.shape[1] * scale_percent)
            height = round(image.shape[0] * scale_percent)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            padd1 = (size - image.shape[1] + 1) // 2
            padd0 = (size - image.shape[0] + 1) // 2
            images.append(cv2.copyMakeBorder(image, padd0, size - image.shape[0] - padd0, 
                padd1, size - image.shape[1] - padd1, cv2.BORDER_CONSTANT, 0))
        return np.array(images)


    def get_crop(self, img, margin, transform = resize_pad_concat):
        # margin is extra space to ensure the hand is included necessarily
        results = self.get_landmarks(img)
        cropped_imgs = []
        for handLms in results:
            lm_x_max = max(lm[0] for lm in handLms)
            lm_y_max = max(lm[1] for lm in handLms)
            lm_x_min = min(lm[0] for lm in handLms)
            lm_y_min = min(lm[1] for lm in handLms)
            cropped_img = img[lm_y_min : lm_y_max, lm_x_min : lm_x_max]
            if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                continue
            cropped_imgs.append(cropped_img)
        return transform(cropped_imgs)









