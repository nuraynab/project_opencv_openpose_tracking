#from __future__ import print_function
import argparse
import cv2
import extractor
import face_recognition
import imutils
import numpy as np
import pickle
import sys

from imutils.object_detection import non_max_suppression
from imutils import paths

def getKeypoints(probMap, threshold=0.1):

	mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

	mapMask = np.uint8(mapSmooth>threshold)
	keypoints = []

	#find the blobs
	_, contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#for each blob find the maxima
	for cnt in contours:
		blobMask = np.zeros(mapMask.shape)
		blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
		maskedProbMap = mapSmooth * blobMask
		_, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
		keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

	return keypoints

# Find valid connections between the different joints of a all persons present
def getValidPairs(output, detected_keypoints):
	valid_pairs = []
	invalid_pairs = []
	n_interp_samples = 10
	paf_score_th = 0.1
	conf_th = 0.7
	# loop for every POSE_PAIR
	for k in range(len(map_Idx)):
		# A->B constitute a limb
		pafA = output[0, map_Idx[k][0], :, :]
		pafB = output[0, map_Idx[k][1], :, :]
		pafA = cv2.resize(pafA, (frameWidth, frameHeight))
		pafB = cv2.resize(pafB, (frameWidth, frameHeight))

		# Find the keypoints for the first and second limb
		candA = detected_keypoints[pose_pairs[k][0]]
		candB = detected_keypoints[pose_pairs[k][1]]
		nA = len(candA)
		nB = len(candB)

		# If keypoints for the joint-pair is detected
		# check every joint in candA with every joint in candB
		# Calculate the distance vector between the two joints
		# Find the PAF values at a set of interpolated points between the joints
		# Use the above formula to compute a score to mark the connection valid

		if( nA != 0 and nB != 0):
			valid_pair = np.zeros((0,3))
			for i in range(nA):
				max_j=-1
				maxScore = -1
				found = 0
				for j in range(nB):
					# Find d_ij
					d_ij = np.subtract(candB[j][:2], candA[i][:2])
					norm = np.linalg.norm(d_ij)
					if norm:
						d_ij = d_ij / norm
					else:
						continue
					# Find p(u)
					interp_coord = list(zip(np.linspace(candA[i][0], candB[j][0], num=n_interp_samples),
											np.linspace(candA[i][1], candB[j][1], num=n_interp_samples)))
					# Find L(p(u))
					paf_interp = []
					for k in range(len(interp_coord)):
						paf_interp.append([pafA[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))],
											pafB[int(round(interp_coord[k][1])), int(round(interp_coord[k][0]))] ])
					# Find E
					paf_scores = np.dot(paf_interp, d_ij)
					avg_paf_score = sum(paf_scores)/len(paf_scores)

					# Check if the connection is valid
					# If the fraction of interpolated vectors aligned with PAF is higher then threshold -> Valid Pair
					if ( len(np.where(paf_scores > paf_score_th)[0]) / n_interp_samples ) > conf_th :
						if avg_paf_score > maxScore:
							max_j = j
							maxScore = avg_paf_score
							found = 1
				# Append the connection to the list
				if found:
					valid_pair = np.append(valid_pair, [[candA[i][3], candB[max_j][3], maxScore]], axis=0)

			# Append the detected connections to the global list
			valid_pairs.append(valid_pair)
		else: # If no keypoints are detected
			print("No Connection : k = {}".format(k))
			invalid_pairs.append(k)
			valid_pairs.append([])
	return valid_pairs, invalid_pairs

# This function creates a list of keypoints belonging to each person
# For each detected valid pair, it assigns the joint(s) to a person
def getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list):
	# the last number in each row is the overall score
	personwiseKeypoints = -1 * np.ones((0, 19))

	for k in range(len(map_Idx)):
		if k not in invalid_pairs:
			partAs = valid_pairs[k][:,0]
			partBs = valid_pairs[k][:,1]
			indexA, indexB = np.array(pose_pairs[k])

			for i in range(len(valid_pairs[k])):
				found = 0
				person_idx = -1
				for j in range(len(personwiseKeypoints)):
					if personwiseKeypoints[j][indexA] == partAs[i]:
						person_idx = j
						found = 1
						break

				if found:
					personwiseKeypoints[person_idx][indexB] = partBs[i]
					personwiseKeypoints[person_idx][-1] += keypoints_list[partBs[i].astype(int), 2] + valid_pairs[k][i][2]

				# if find no partA in the subset, create a new subset
				elif not found and k < 17:
					row = -1 * np.ones(19)
					row[indexA] = partAs[i]
					row[indexB] = partBs[i]
					# add the keypoint_scores for the two keypoints and the paf_score
					row[-1] = sum(keypoints_list[valid_pairs[k][i,:2].astype(int), 2]) + valid_pairs[k][i][2]
					personwiseKeypoints = np.vstack([personwiseKeypoints, row])
	return personwiseKeypoints

def get_args():
	ap = argparse.ArgumentParser()

	ap.add_argument("-e", "--encodings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-i", "--input", required=True, 
		help="path to input video")
	ap.add_argument("-o", "--output", type=str,
		help="path to output video")
	ap.add_argument("-d", "--detection-method", type=str, default="cnn",
		help="face detection model to use: either `hog` or `cnn`")
	ap.add_argument("-t", "--type", type=str, default="video",
		help="input type: either video or frames")
	args = vars(ap.parse_args())
	return args

def recognize_faces(frame, width, height):

	caffemodel = {"prototxt":"./deploy.prototxt",
			  "model":"./res10_300x300_ssd_iter_140000.caffemodel",
			  "acc_threshold":0.50
			  }

	net = cv2.dnn.readNetFromCaffe(caffemodel["prototxt"], caffemodel["model"])
	(H, W) = (None, None)  		# input image height and width for the network

	# # grab the next frame
	# (grabbed, frame) = stream.read()

	# if the frame was not grabbed, then we have reached the
	# end of the stream
	# if not grabbed:
	# 	print('Failed to read video')
	# 	sys.exit(1)

	image = frame[:, :, ::-1]
	image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
	if W is None or H is None: (H, W) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()	# detect objects using object detection model
	detections_bbox = []		# bounding box for detections
	detections_bbox2 = []		# has another formatting
	names = []

	for i in range(0, detections.shape[2]):
		if detections[0, 0, i, 2] > caffemodel["acc_threshold"]:
			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H ])
			detections_bbox2.append(box.astype("int"))

	for [startX, startY, endX, endY] in detections_bbox2:
		detections_bbox.append(tuple((startY, endX, endY, startX)))

	encodings = face_recognition.face_encodings(image, detections_bbox)

	with open("aaron_logger", "a") as file:
		file.write("frame {}: {} \n".format(frame_number, encodings))

	# loop over the facial embeddings
	for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
		matches = face_recognition.compare_faces(data["encodings"],
			encoding, tolerance = 0.4)
		name = "Unknown"

		# check to see if we have found a match
		if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# loop over the matched indexes and maintain a count for
			# each recognized face face
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
			name = max(counts, key=counts.get)
		
		# update the list of names
		names.append(name)
	return detections_bbox2, names

def detect_bodies(frame, pose_pairs, map_Idx):

	proto_File = "pose_deploy_linevec.prototxt"
	weights_File = "pose_iter_440000.caffemodel"
	nPoints = 18

	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]

	net = cv2.dnn.readNetFromCaffe(proto_File, weights_File)

	# Fix the input Height and get the width according to the Aspect Ratio
	inHeight = 368
	inWidth = int((inHeight/frameHeight)*frameWidth)

	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
							(0, 0, 0), swapRB=False, crop=False)

	net.setInput(inpBlob)
	output = net.forward()

	detected_keypoints = []
	keypoints_list = np.zeros((0,3))
	keypoint_id = 0
	threshold = 0.1

	for part in range(nPoints):
		probMap = output[0,part,:,:]
		probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))
		keypoints = getKeypoints(probMap, threshold)
		#print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))
		keypoints_with_id = []
		for i in range(len(keypoints)):
			keypoints_with_id.append(keypoints[i] + (keypoint_id,))
			keypoints_list = np.vstack([keypoints_list, keypoints[i]])
			keypoint_id += 1

		detected_keypoints.append(keypoints_with_id)


	frameClone = frame.copy()

	valid_pairs, invalid_pairs = getValidPairs(output, detected_keypoints)
	personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, keypoints_list)

	body_boxes = []
	openpose_faces = []

	for i in range(19):
		if i != 17 and i != 12:
			continue
		if i == 17:
			for n in range(len(personwiseKeypoints)):
				index = personwiseKeypoints[n][np.array(pose_pairs[i])]
				if -1 in index:
					continue
				B = np.int32(keypoints_list[index.astype(int), 0])
				A = np.int32(keypoints_list[index.astype(int), 1])
				#cv2.rectangle(frameClone, (B[0], A[0]), (B[1], A[1]), (0, 255, 0), 2)
				(X1, Y1, X2, Y2) = (B[0], A[0], B[1], A[1])
				body_boxes.append((X1 - int((X2 - X1)/2), Y1, X2 + int((X2 - X1)/2), Y2))
		if i == 12:
			for n in range(len(personwiseKeypoints)):
				index = personwiseKeypoints[n][np.array(pose_pairs[i])]
				if -1 in index:
					continue
				B = np.int32(keypoints_list[index.astype(int), 0])
				A = np.int32(keypoints_list[index.astype(int), 1])
				#cv2.rectangle(frameClone, (B[0], A[0]), (B[1], A[1]), (0, 255, 0), 2)
				(X2, Y2) = (B[1], A[1])
				openpose_faces.append((X2, Y2))
	return body_boxes, openpose_faces

if __name__ == '__main__':
  
	tracker_Type = "KCF"

	args = get_args()


	pose_pairs = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
			  [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
			  [1,0], [0,14], [14,16], [0,15], [15,17],
			  [2,13], [5,16]]

	# index of pafs correspoding to the pose_pairs
	# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.
	map_Idx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44],
				[19,20], [21,22], [23,24], [25,26], [27,28], [29,30],
				[47,48], [49,50], [53,54], [51,52], [55,56],
				[37,38], [45,46]]

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(args["encodings"], "rb").read())

	# initialize the pointer to the video file and the video writer
	print("[INFO] processing video...")
	input_type = args["type"]
	if input_type == "video":
		stream = cv2.VideoCapture(args["input"])
		length = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
		(grabbed, frame) = stream.read()
		if not grabbed:
			print('Failed to read video')
			sys.exit(1)
	else:
		imagePaths = list(paths.list_images(args["input"]))
		frame = cv2.imread(imagePaths[0])
		length = len(imagePaths)
	writer = None
	frame_number=0

	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]

	#width = int(stream.get(3))  # float
	#height = int(stream.get(4)) # float

	
	multiTracker = cv2.MultiTracker_create()
	multiTracker_body = cv2.MultiTracker_create()
	face_boxes, names = recognize_faces(frame, frameWidth, frameHeight)
	body_boxes, openpose_faces = detect_bodies(frame, pose_pairs, map_Idx)

	# Initialize MultiTracker 
	for bbox in face_boxes:
		multiTracker.add(cv2.TrackerKCF_create(), frame, tuple(bbox))

	# Initialize MultiTracker 
	for body_box in body_boxes:
		multiTracker_body.add(cv2.TrackerKCF_create(), frame, body_box)

	while True:
		frame_number += 1
		if input_type == "video":
			# grab the next frame
			(grabbed, frame) = stream.read()

			# if the frame was not grabbed, then we have reached the
			# end of the stream
			if not grabbed:
				break
		else:
			frame = cv2.imread(imagePaths[frame_number])

		if frame_number % 5 == 0:
			multiTracker = cv2.MultiTracker_create()
			multiTracker_body = cv2.MultiTracker_create()
			face_boxes, names = recognize_faces(frame, frameWidth, frameHeight)
			body_boxes, openpose_faces = detect_bodies(frame, pose_pairs, body_boxes)

			# Initialize MultiTracker 
			for bbox in face_boxes:
				multiTracker.add(cv2.TrackerKCF_create(), frame, tuple(bbox))

			# Initialize MultiTracker 
			for body_box in body_boxes:
				multiTracker_body.add(cv2.TrackerKCF_create(), frame, body_box)
		else:
			# get updated location of objects in subsequent frames
			grabbed, face_boxes = multiTracker.update(frame)
			grabbed, body_boxes = multiTracker_body.update(frame)
		

		# loop over the recognized faces
		for ((left, top, right, bottom), name) in zip(face_boxes, names):
			top = int(top)
			right = int(right)
			bottom = int(bottom)
			left = int(left)

			for face_points in openpose_faces:
				if face_points[0] > left and face_points[0] < right:
					j = openpose_faces.index(face_points)

					if j >= len(body_boxes):
						continue

					(body_left, body_top, body_right, body_bottom) = body_boxes[j]
					body_top = int(body_top)
					body_right = int(body_right)
					body_bottom = int(body_bottom)
					body_left = int(body_left)

					if name == "Unknown":
						try:							
							cv2.rectangle(frame, (body_left, body_top), (body_right, body_bottom), (0, 255, 0), 2)
							sub_face = frame[body_top:body_bottom, body_left:body_right]
							sub_face = cv2.GaussianBlur(sub_face,(23, 23), 70)
							frame[body_top:body_top + sub_face.shape[0], body_left:body_left + sub_face.shape[1]] = sub_face

						except AttributeError:
							print("shape not found")

					else:
						cv2.rectangle(frame, (body_left, body_top), (body_right, body_bottom), (0, 255, 0), 2)
						y = body_top - 15 if body_top - 15 > 15 else body_top + 15
						cv2.putText(frame, name, (body_left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

			# draw the predicted face name on the image
			if name == "Unknown":
				try:
					print(left, top, right, bottom)
					cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
					sub_face = frame[top:bottom, left:right]
					if len(sub_face) == 0:
						print("len sub_face == 0")
						continue
					else:
						sub_face = cv2.GaussianBlur(sub_face,(23, 23), 70)
						print("blur face")
						frame[top:top + sub_face.shape[0], left:left + sub_face.shape[1]] = sub_face
				except AttributeError:
					print("shape not found")
				
			else:
				print((left, top, right, bottom))
				cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
				print("recognized face")
				y = top - 15 if top - 15 > 15 else top + 15
				cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
					0.75, (0, 255, 0), 2)
				with open("aaron_logger", "a") as file:
					file.write("frame {}: {} \n".format(frame_number, (left, top, right, bottom)))

		# if the writer is not None, write the frame with recognized
		# faces t odisk
		if writer is not None:
			print("Writing frame {} / {}".format(frame_number, length))
			writer.write(frame)

		# if the video writer is None *AND* we are supposed to write
		# the output video to disk initialize the writer
		if writer is None and args["output"] is not None:
			print("Writing frame {} / {}".format(frame_number, length))
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 24,
			(frame.shape[1], frame.shape[0]), True)

	# close the video file pointers
	stream.release()

	# check to see if the video writer point needs to be released
	if writer is not None:
		writer.release()
