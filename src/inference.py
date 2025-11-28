from ultralytics import YOLO
from pathlib import Path
import torch
from PIL import Image
from src.config import load_config

class YOLOv11_inference:
    def __init__(self, model_name, device = 'cuda'):
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(device)
        
        # Loading configurations from default.yaml
        config = load_config()
        self.conf_threshold = config["model"]["conf_threshold"]
        self.extensions = config["data"]["image_extension"]
    
    
    # Now a process image method which will take the image path as input and return the image 
    def process_image(self,image_path):
        # Running the inference on the image 
        results = self.model.predict(
            source = image_path,
            conf = self.conf_threshold,
            device = self.device
        )
        
        # Process the results
        detection = [] # It is the list of dictionaries
        class_counts = {} # All the inferenced image we get as result, also returns the class count.
        
        for result in results:
            # Boxes in the yolo detection model have class{which is an int}, confidence score, bbox co-ordinates
            for box in result.boxes:
                cls = result.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()
                
                detection.append({
                    'class':cls,
                    'confidence':conf,
                    'bbox':bbox,
                    'count':1
                })
                
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        for det in detection:
            det['count'] = class_counts[det['class']]
        
        return {
            'image_path': str(image_path),
            'detections': detection,
            'total_objects': len(detection),
            'unique_classes': list(class_counts.keys()),# [0,1,2....] --> classes
            'class_counts': class_counts # {0:1, 1:10 ...} ---> counts of each class in each image
        }
        
    
    # This method will be responsible for taking the directory and getting the image from it to process
    def process_dir(self, directory):
        metadata = [] # This is the metadata list which will contain all the information about the images
        
        patterns = [f"*{ext}" for ext in self.extensions]
        
        # Now we will use these patterns for getting all the files of a particular pattern
        image_path = []
        for pattern in patterns:
            image_path.extend(Path(directory).glob(pattern))
        
        # Traversing each and every image through image path
        for img_path in image_path:
            try:
                metadata.append(self.process_image(img_path))
            except Exception as e:
                print(f"Error processing the image...{img_path}: {str(e)}")
                continue
        
        return metadata