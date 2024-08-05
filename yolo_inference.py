from ultralytics import YOLO 

model = YOLO('models/best.pt')

results = model.predict(r"C:\Users\44780\Documents\motion_tracking_project\input_videos\08fd33_4.mp4",save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)