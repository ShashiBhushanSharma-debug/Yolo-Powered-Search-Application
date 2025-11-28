import streamlit as st
import sys
import time
from pathlib import Path
from src.inference import YOLOv11_inference
from src.utils import save_metadata, load_metadata, get_unique_classes_counts
from PIL import Image, ImageDraw, ImageFont
from src.utils import img_to_base64
import json

# Adding the project to the system path
sys.path.append(str(Path(__file__).parent))

def init_session_state():
    # This function is going to save the session state info in terms of rerun of the streamlit application
    session_defaults = {
        'metadata': None,
        'unique_classes': [],
        'count_options': {},
        "search_results": [],
        'show_bbox': True,
        'grid_columns': 3,
        'highlight_classes': False,
        "search_params": {
            "search_mode": "Any of selected (OR)",
            "selected_classes": [],
            "thresholds": {}
        }
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

st.set_page_config(page_title="YOLOv11 Search App", layout="wide")
st.title("Computer Vision Powered Search Application")

# Custom CSS for perfect grid layout
st.markdown(f"""
<style>
/* Main container adjustments */
.st-emotion-cache-1v0mbdj {{
    width: 100% !important;
    height: 100% !important;
}}

/* Column container - critical for grid layout */
.st-emotion-cache-1wrcr25 {{
    max-width: none !important;
    padding: 0 1rem !important;
}}

/* Individual column styling */
.st-emotion-cache-1n76uvr {{
    padding: 0.5rem !important;
}}

/* Image cards */
.image-card {{
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    margin-bottom: 20px;
    background: #f8f9fa;
}}

.image-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.15);
}}

.image-container {{
    position: relative;
    width: 100%;
    aspect-ratio: 4/3;
}}

.image-container img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
}}

.meta-overlay {{
    padding: 10px;
    background: rgba(0,0,0,0.85);
    color: white;
    font-size: 13px;
    line-height: 1.4;
}}
</style>
""", unsafe_allow_html=True)

# Main Options in the application
option = st.radio("Choose an option:- ", ("Process new images", "Load excisting metadata"), horizontal=True)

# This section will be responsible for the processing of the image and saving the metadata of the images.

if option == "Process new images":
     with st.expander("Process new images", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            image_dir = st.text_input("Image directory path:- ", placeholder="path_to_images")
        with col2:
            model_path = st.text_input("Model weight path:- ", "yolo11m.pt") # This is the place where we mention the model name
        if st.button("Start inferencing"):
            if image_dir:
                try:
                    with st.spinner("Running object detection..."):
                        inferencer = YOLOv11_inference(model_path)
                        metadata = inferencer.process_dir(image_dir)
                        # Do something with the list 
                        # We need to save the metadata and also load the classes and class counts using this meta data
                        metadata_path = save_metadata(metadata, image_dir)
                        st.success(f"Processed {len(metadata)} images. Metadata saved to: ")
                        st.code(str(metadata_path))
                        st.session_state.metadata = metadata
                        st.session_state.unique_classes, st.session_state.count_options =  get_unique_classes_counts(metadata)
                        print(f"{st.session_state.unique_classes}, {st.session_state.count_options}")
                        
                except Exception as e:
                    st.error(f"Found error during inference: {str(e)}")
            else:
                st.warning("Enter an image directory path.")
        
else:
    with st.expander("Load existing Metadata", expanded=True):
        metadata_path = st.text_input("Metadata file path:", placeholder="path/to/metdata.json")
        if st.button("Load Metadata"):
            if metadata_path:
                try:
                   with st.spinner("Loading the Metadata..."):
                    metadata = load_metadata(metadata_path)
                    st.session_state.metadata = metadata
                    st.session_state.unique_classes, st.session_state.count_options =  get_unique_classes_counts(metadata)
                    print(f"{st.session_state.unique_classes}, {st.session_state.count_options}")
                    st.success("Successfully loaded the metadata")
                except Exception as e:
                    st.error(f"Error loading the Metadata {str(e)}")
            else:
                st.warning("Give the path to the metadata.")
                
# st.write(f"{st.session_state.unique_classes}, {st.session_state.count_options}")

# Search Functionality

if st.session_state.metadata:
    st.header("Search Engine")
    
    # We need to create this container to fill all the elements 
    with st.container():
        st.session_state.search_params["search_mode"] = st.radio("Select the search mode: ", ("Any of selected (OR)", "All selected class(AND)"), horizontal=True) 
        
        st.session_state.search_params["selected_classes"] = st.multiselect("Classes to Search for:- ",
                       options= st.session_state.unique_classes
                       )
        
        if st.session_state.search_params["selected_classes"]:
            st.subheader("Count Threshold (optional): ")
            cols = st.columns(len(st.session_state.search_params["selected_classes"]))
            
            # Now we will iterate over the selected classes to assign a threshold to it one by one
            for i, cls in enumerate(st.session_state.search_params["selected_classes"]):
                with cols[i]:
                    st.session_state.search_params["thresholds"][cls] = st.selectbox(
                        f"Max count for the {cls}",
                        options= ["None"] + st.session_state.count_options[cls],
                        
                    )
        
        if st.button(type="primary", label="Search image") and st.session_state.search_params["selected_classes"]:
            result = []
            search_params = st.session_state.search_params
            
            for items in st.session_state.metadata:
                matches = False
                class_matches = {}
                
                for cls in search_params["selected_classes"]:
                    class_detection = [d for d in items['detections'] if d['class'] == cls]
                    class_counts = len(class_detection)
                    class_matches[cls] = False
                    
                    threshold = search_params['thresholds'].get(cls, "None")
                    if threshold == "None":
                        class_matches[cls] = (class_counts>=1)
                    else:
                        class_matches[cls] = (class_counts >= 1 and class_counts<=int(threshold))
                        
                if search_params["search_mode"] == "Any of selected (OR)":
                    # Not work only when both are not present or both the classes are false --> using the function any()
                    matches = any(class_matches.values())
                else:
                    # This is for the AND mode
                    # Only work when both are true
                    matches = all(class_matches.values())

                if matches:
                    result.append(items)
                    
            st.session_state.search_results = result
        
# Display the results
if st.session_state.search_results:
    results = st.session_state.search_results
    search_params = st.session_state.search_params
    
    st.subheader(f"Result {len(st.session_state.search_results)} matching images")
    
    # Display Controls
    with st.expander("Display options", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.show_bbox = st.checkbox("Show bounding boxes")
            value = st.session_state.show_bbox
        with col2:
            st.session_state.grid_columns = st.slider("Grid Columns", 3, 6)
            value = st.session_state.grid_columns
        with col3:
            st.session_state.highlight_classes = st.checkbox("highlight classes")
            value = st.session_state.highlight_classes
            
    # Creating grid using streamlit columns
    grid_columns = st.columns(st.session_state.grid_columns)
    col_index = 0
    
    for result in results:
        with grid_columns[col_index]:
            #try:
                # Now we need to show the images and we have image path in the meta data so, we will use Image from pillow
                img = Image.open(result['image_path'])
                draw= ImageDraw.Draw(img)
                
                if st.session_state.show_bbox:
                    try:
                        font = ImageFont.truetype("arial.ttf", 12)
                    except Exception as e:
                        font = ImageFont.load_default()
                        
                    for det in result['detections']:
                        cls = det['class']
                        bbox = det['bbox']
                        
                        if cls in search_params["selected_classes"]:
                            color = "#19C70A"
                            thickness = 3
                        elif not st.session_state.highlight_classes:
                            color = "#666666"
                            thickness = 1
                        else:
                            continue
                        
                        draw.rectangle(bbox, outline=color, width=thickness)
                        
                        if cls in search_params['selected_classes'] or not st.session_state.highlight_classes:
                            label = f"{cls} {det['confidence']:.3f}"
                            text_bbox = draw.textbbox((0,0), label, font=font)
                            width_x = text_bbox[2] - text_bbox[0]
                            width_y = text_bbox[3] - text_bbox[1]
                            
                            draw.rectangle([bbox[0], bbox[1],bbox[0] + width_x + 8, bbox[1] + width_y+4],
                                        fill = color)
                            draw.text((bbox[0] + 4, bbox[1] + 2),
                                    label,
                                    fill="white",
                                    font=font
                                    ) 
                            
                meta_items = [f"{key}: {val}" for key, val in result['class_counts'].items() if key in st.session_state.search_params['selected_classes']]         
                
                # Display card for the image
                st.markdown(f"""
                    <div class = "image_card">
                        <div class = "image_container">
                            <img src="data:image/png;base64,{img_to_base64(img)}">
                            </div>
                            <div class="meta-overlay">
                                <strong>{Path(result['image_path']).name}</strong><br>
                                {", ".join(meta_items) if meta_items else "No matches"}
                            </div>
                        </div>    
                            """, unsafe_allow_html=True
                    )
                
            # except Exception as e:
            #     st.error(f"Error displaying the image {result['image_path']} : {str(e)}")
        
        col_index = (col_index + 1) % st.session_state.grid_columns
        
    with st.expander("Export Options"):
        st.download_button(
            label="Download Results (JSON)",
            data = json.dumps(results,indent=2),
            file_name="search_results.json",
            mime="application/json"
        )
        