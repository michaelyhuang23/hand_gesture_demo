from hand_detector import HandDetector
import cv2
import numpy as np
import time

handDetector = HandDetector()


cap = cv2.VideoCapture(0)

width  = int(cap.get(3))
height = int(cap.get(4))
print(f'width: {width}, height: {height}')
drawpad = np.zeros((height, width, 3),dtype=np.uint8)

ptr_trace = []
while cap.isOpened():
	ret, frame = cap.read()
	lmks = handDetector.get_landmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	if lmks is None:
		continue
	for handLmks in lmks:
		for lm in handLmks:
			frame = cv2.circle(frame, (lm[0],lm[1]), radius=10, color=(255, 0, 0), thickness=-1)

	cmd = cv2.waitKey(1) & 0xFF
	if cmd == ord('w'):
		for i, handLmks in enumerate(lmks):
			if len(ptr_trace)<=i:
				ptr_trace.append((handLmks[8][0], handLmks[8][1],time.time()))
			else: 
				if time.time() - ptr_trace[i][2] < 0.3:
					print(ptr_trace[i][0:2], handLmks[2][0:2])
					drawpad  = cv2.line(drawpad, ptr_trace[i][0:2], tuple(handLmks[8][0:2]), (0,255,255), 20)
				ptr_trace[i] = (handLmks[8][0], handLmks[8][1],time.time())
	elif cmd == ord('c'):
		drawpad = np.zeros((height, width, 3),dtype=np.uint8)
		ptr_trace = []
	elif cmd == ord('q'):
		break

	frame = np.where(drawpad > 0, drawpad, frame)
	cv2.imshow('win', frame[:,::-1,:])



