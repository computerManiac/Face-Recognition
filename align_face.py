import dlib
import cv2
import openface

model = "shape_predictor_68_face_landmarks.dat"

face_detector = dlib.get_frontal_face_detector()
face_pose_detector = dlib.shape_predictor(model)
face_aligner = openface.AlignDlib(model)

faces = []

def align(image):

	#image = cv2.imread(image_name)

	detected_faces = face_detector(image,1)
	print("Found {} faces in the given image".format(len(detected_faces)))

	for i,face_Rect in enumerate(detected_faces):

		landmarks = face_pose_detector(image,face_Rect)
		aligned_faces = face_aligner.align(96,image,face_Rect,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

		faces.append(aligned_faces)

	return faces,detected_faces


