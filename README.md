# model source
the modole resource: https://huggingface.co/Ultralytics/YOLOv8/tree/main
YOLOV8 documentation: https://docs.ultralytics.com/reference/engine/results/
YOLOV11 model: https://github.com/ultralytics/ultralytics?tab=readme-ov-file

# check the web camera on linux computer 
```bash
v4l2-ctl --list-devices
```

# time measurement
run the yolov5n.pt on linux computer: 
```average model inference time is:  0.012```
run the yolov8n.pt on linux computer:
```average model inference time is:  0.011``` -- current best
run the yolo11n.pt on linux computer:
```average model inference time is:  0.016``` -- current best
run the yolov1 VGG19-BN on linux compiuter 
```average model inference time is 1.96 second``` --> bad
