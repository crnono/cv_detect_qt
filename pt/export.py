from ultralytics import YOLO

# Load a model
model = YOLO("yolo11x.pt")
path = model.export(format="onnx")
