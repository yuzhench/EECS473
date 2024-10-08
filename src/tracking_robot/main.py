import torch
import cv2
from ultralytics import YOLO
from util import Tracker
import random
import json
import argparse
#lib for clicking the img 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

clicked_x = None
clicked_y = None
target_id = None
first_frame = True

def if_in_bbox (x1, y1, x2, y2):
    global clicked_x, clicked_y
    if clicked_x > x1 and clicked_x < x2 and clicked_y > y1 and clicked_y < y2:
        return True
    else:
        return False
    

def onclick(event):
    global clicked_x
    global clicked_y
    if event.xdata is not None and event.ydata is not None:
        clicked_x = int(event.xdata)
        clicked_y = int(event.ydata)
        print(f'x = {int(clicked_x)}, y = {int(clicked_y)}')
        plt.close()

def setup_arguments():
    parser = argparse.ArgumentParser(description="choose the mode ")

    parser.add_argument('-c', '--choose', type=bool, default=True)    
    return parser


def load_color (json_path):
    with open(json_path, 'r') as file:
        colors = json.load(file)

    return colors

def choose_the_person(frame):
     #conver to RGB camera 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #ready to click the frame
    fig, ax = plt.subplots()
    ax.imshow(frame)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    

def process_video(input_video_path, output_video_path, model, tracker, colors, args):
    #define the globle variable 
    global first_frame
    global clicked_x, clicked_y
    global target_id
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
        
        print("the first_frame is: ", first_frame)
        #first frame: choose the target 
        if first_frame == True:
            choose_the_person(frame)
            first_frame = False
          
            

        #for the rest of the frame: tracking the target
        elif first_frame == False:
            if (clicked_x is not None ) and (clicked_y is not None): 
                # Process frame through the model
                results = model(frame)
                boxes = results[0].boxes

                detections = []
                for i, (curr_xyxy, conf, cls_id) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):   
                
                    x1, y1, x2, y2 =  int(curr_xyxy[0]), int(curr_xyxy[1]), int(curr_xyxy[2]), int(curr_xyxy[3])
                    #define detection:
                    if model.names[int(cls_id)] == 'person':
                        detections.append([x1,y1,x2,y2,conf.item()])

                
        
                #try to update the tracker 
                tracker.update(frame, detections)
                print("successfully update the tracker")


                
                for track in tracker.tracks:
                    label = 'person'
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id
                    #store the target info to global 
                    if target_id == None and if_in_bbox(x1, y1, x2, y2,):
                        target_id = track_id

                    if args.choose == True:
                        if track_id == target_id:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 2)
                            cv2.putText(frame, f'{label} {track_id} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
                        else:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                            # cv2.putText(frame, f'{label} {track_id} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

                    else:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 2)
                        cv2.putText(frame, f'{label} {track_id} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)


        


        # Write the frame into the output file
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()




def main():

    parser = setup_arguments()
    args = parser.parse_args()
    # Load the YOLOv8 model
    model = YOLO('yolov8x.pt')

    tracker = Tracker()

    #generate the color
    colors = load_color('color.json')
    # Process the video
    process_video('../../video/p.mp4', 'output_video_with_tracking_choose.mp4', model, tracker, colors, args)



if __name__ == "__main__": 
    main()