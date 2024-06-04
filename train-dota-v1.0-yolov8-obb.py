from ultralytics import YOLO

def train():
    model = YOLO('yolov8n-obb-custom.yaml').load('yolov8n-obb.pt')
    model.train(data='DOTAv1-custom.yaml', epochs=100, imgsz=1024, batch=4, workers=4)

def validation():
    model = YOLO(r'runs/obb/train/weights/best.pt')
    model.val(data='dotav1.yaml', imgsz=1024, batch=4, workers=4)

def inference():
    from ultralytics import YOLO
    model = YOLO('runs/obb/train/weights/best.pt')
    results = model('datasets/DOTA/images/val/P0003__1__0___0.png', save=True)


if __name__ == "__main__":
    train()