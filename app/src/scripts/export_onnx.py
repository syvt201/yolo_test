from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")  # load an official model

with open("classes.txt", "w") as f:
    for cls_id, cls in model.names.items():
        f.write(f"{cls_id},{cls}\n")
# Export the model
model.export(format="onnx", nms=False)