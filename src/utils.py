"""
We are creating this utils.py file to save the processed metadata from inference to the system.

"""
import base64 # To embedd the image into the HTML code section
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import io

def ensure_process_dir_present(raw_path):
    raw_path = Path(raw_path)
    processed_path = raw_path.parent.parent / "processed" / raw_path.name
    processed_path.mkdir(parents=True, exist_ok=True)
    return processed_path

def save_metadata(metadata, raw_path):
    pro_path = ensure_process_dir_present(raw_path)
    output_path = pro_path / "metadata.json"
    with open(output_path, 'w') as f:
        json.dump(metadata, f)
    return output_path

def load_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    
    # There can be the possibility that the user has input the wrong json path
    if not metadata_path.exists():
        processed_path = metadata_path.parent.parent / "processed" / str(metadata_path.name) / "metadata.json"
        if processed_path.exists():
            metadata_path = processed_path
        else:
            raise FileNotFoundError(f"metadata is not found at location {metadata_path}")
    # Load the metadata path
    with open(metadata_path, 'r') as f:
        return json.load(f)

def get_unique_classes_counts(metadata):
       df = pd.DataFrame(metadata)
       unique_classes = set()
       count_options = {}
       
       for item in metadata:
           for cls in item['detections']:
                unique_classes.add(cls['class'])
                if cls['class'] not in count_options:
                   count_options[cls['class']] = set()
                count_options[cls['class']].add(cls['count'])
    
       unique_classes = sorted(unique_classes)
       for cls in count_options:
           count_options[cls] = sorted(count_options[cls])
           
       return unique_classes, count_options 

def img_to_base64(image : Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
        