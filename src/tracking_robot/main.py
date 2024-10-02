import torch
import cv2
from ultralytics import YOLO
from util import Tracker
import random
import json
import argparse



def setup_arguments():
    parser = argparse.ArgumentParser(description="choose the mode ")

    parser.add_argument('-c', '--choose', type=bool, default=True)
    parser.add_argument('-t', '--toggle', action='store_true', help="Toggle some feature on.")
    
    return parser


def load_color (json_path):
    with open(json_path, 'r') as file:
        colors = json.load(file)

    return colors

def process_video(input_video_path, output_video_path, model, tracker, colors):
    # Open the video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame through the model
        results = model(frame)

        # print(results[0])

        # exit(0)
 
        boxes = results[0].boxes
        # print(boxes)

        # exit(0)

         
 
        detections = []

        for i, (curr_xyxy, conf, cls_id) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls))  :  # detections per frame
           
          
            x1, y1, x2, y2 =  int(curr_xyxy[0]), int(curr_xyxy[1]), int(curr_xyxy[2]), int(curr_xyxy[3])

            #define detection:
            if model.names[int(cls_id)] == 'person':
                detections.append([x1,y1,x2,y2,conf.item()])

        
        # print(detections)
    
        # exit(0)
        #try to update the tracker 
        tracker.update(frame, detections)
        print("successfully update the tracker")


        
        for track in tracker.tracks:
            label = 'person'
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 2)
            cv2.putText(frame, f'{label} {track_id} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)


        


        # Write the frame into the output file
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()




# Load the YOLOv8 model
model = YOLO('yolov8x.pt')

tracker = Tracker()

#generate the color
colors = load_color('color.json')
# Process the video
process_video('../../video/p.mp4', 'output_video_with_tracking.mp4', model, tracker, colors)
