import os
import json
import base64
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from utils import save_template, load_template
from alignment import align_images, AlignmentError
from extractor import OCRExtractor

# 1. Page Configuration & Theme Styling
st.set_page_config(
    page_title="Configuration-Driven OCR Engine",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for rich aesthetics
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap');
    
    .main {
        font-family: 'Outfit', sans-serif;
    }
    
    .header-container {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .header-container h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    
    .header-container p {
        font-size: 1.1rem;
        margin-top: 5px;
        opacity: 0.9;
        color: #e0e0e0;
    }
    
    .footer-container {
        text-align: center;
        margin-top: 50px;
        padding: 15px;
        font-size: 0.9rem;
        color: #777;
        border-top: 1px solid #eee;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #ddd;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="header-container">
        <h1>Configuration-Driven OCR Engine</h1>
        <p>Define template zones on reference files and extract data from skewed documents using CV Homography and EasyOCR</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Lazy-loaded and cached OCR Extractor instance
@st.cache_resource
def get_ocr_extractor():
    return OCRExtractor(gpu=True)

# Helper to draw bounding boxes in RGB
def draw_template_boxes(image: np.ndarray, fields: list, current_clicks: list) -> np.ndarray:
    if image is None:
        return None
    
    draw_img = image.copy()
    
    # 1. Draw saved fields (Green box and label text)
    for field in fields:
        x = int(field['x'])
        y = int(field['y'])
        w = int(field['w'])
        h = int(field['h'])
        label = field['label']
        
        cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 200, 0), 2)
        # Background block for text label
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(draw_img, (x, y - text_size[1] - 8), (x + text_size[0] + 6, y), (0, 200, 0), -1)
        cv2.putText(draw_img, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
    # 2. Draw current selection clicks/box (Red circle or box)
    if len(current_clicks) == 1:
        x1, y1 = current_clicks[0]
        cv2.circle(draw_img, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(draw_img, (x1, y1), 10, (255, 255, 255), 2)
    elif len(current_clicks) == 2:
        x1, y1 = current_clicks[0]
        x2, y2 = current_clicks[1]
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
    return draw_img

# Streamlit Session State Initialization
if "clicks" not in st.session_state:
    st.session_state.clicks = []
if "fields" not in st.session_state:
    st.session_state.fields = []
if "last_click" not in st.session_state:
    st.session_state.last_click = None
if "saved_template" not in st.session_state:
    st.session_state.saved_template = None
if "active_template" not in st.session_state:
    st.session_state.active_template = None
if "key_counter" not in st.session_state:
    st.session_state.key_counter = 0

def clear_selection():
    st.session_state.clicks = []
    st.session_state.last_click = None
    st.session_state.key_counter += 1

def clear_all_fields():
    st.session_state.fields = []
    st.session_state.saved_template = None
    st.session_state.active_template = None
    clear_selection()

# Tab creation
tab1, tab2 = st.tabs(["🏗️ Template Builder", "📄 Extract OCR"])

# ----------------- TAB 1: TEMPLATE BUILDER -----------------
with tab1:
    st.header("Template Builder")
    st.write("Upload a reference document to define the regions of interest (fields) you want to extract.")
    
    # Sidebar/Upload Row
    col_upload, col_actions = st.columns([2, 1])
    with col_upload:
        uploaded_ref = st.file_uploader("Upload Reference Image", type=["png", "jpg", "jpeg"], key="ref_uploader")
        
        # Determine current ref image
        ref_img = None
        ref_name = None
        if uploaded_ref is not None:
            file_bytes = np.asarray(bytearray(uploaded_ref.read()), dtype=np.uint8)
            ref_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_name = uploaded_ref.name
        else:
            # Fallback to test_ref.png if it exists
            default_ref_path = "test_ref.png"
            if os.path.exists(default_ref_path):
                ref_img = cv2.imread(default_ref_path)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_name = default_ref_path
                st.info("💡 Using default reference image (`test_ref.png`).")
        
        # Reset state if reference image changes
        if ref_name:
            if "current_ref_name" not in st.session_state or st.session_state.current_ref_name != ref_name:
                st.session_state.current_ref_name = ref_name
                clear_all_fields()
    
    if ref_img is not None:
        orig_h, orig_w = ref_img.shape[:2]
        
        # Layout splits: Selection Area vs. Fields & Saving
        col_canvas, col_fields = st.columns([3, 2])
        
        with col_canvas:
            st.subheader("1. Click Bounding Boxes on Image")
            
            # Guidelines helper text
            if len(st.session_state.clicks) == 0:
                st.info("📍 Click on the image below to select the **Top-Left** corner of the bounding box.")
            elif len(st.session_state.clicks) == 1:
                st.info(f"📍 Top-Left selected at {st.session_state.clicks[0]}. Now click the **Bottom-Right** corner.")
            elif len(st.session_state.clicks) == 2:
                st.success(f"📏 Corners selected: {st.session_state.clicks[0]} to {st.session_state.clicks[1]}. Enter a label name on the right.")
            
            st.caption("🖱️ **Tip**: Use your scroll wheel (or hold **Shift + Scroll** to scroll horizontally) to move around large template images.")
            
            # Inject CSS for scrollable canvas container
            st.html("""
                <style>
                    .st-key-scrollable_canvas_container {
                        max-height: 600px;
                        overflow-y: auto !important;
                        overflow-x: auto !important;
                        border: 1px solid #ddd;
                        border-radius: 8px;
                        padding: 10px;
                        background-color: #fdfdfd;
                    }
                    /* Ensure custom iframe element behaves within scroll boundaries */
                    .st-key-scrollable_canvas_container iframe {
                        max-width: none !important;
                        display: block;
                    }
                </style>
            """)
            
            # Prepare image with drawn boxes/clicks
            annotated_img = draw_template_boxes(ref_img, st.session_state.fields, st.session_state.clicks)
            
            # Display image and capture click coordinates
            # Fit image on screen using a fixed display width (700px)
            display_width = 700
            display_height = int(display_width * (orig_h / orig_w))
            
            with st.container(key="scrollable_canvas_container"):
                coords = streamlit_image_coordinates(
                    annotated_img,
                    width=display_width,
                    key=f"coordinate_clicker_{st.session_state.key_counter}"
                )
            
            # Handle coordinate selection
            if coords is not None:
                click_key = (coords["x"], coords["y"])
                if st.session_state.last_click != click_key:
                    st.session_state.last_click = click_key
                    # Scale clicked coordinates back to the original image dimensions
                    orig_click_x = int(coords["x"] * (orig_w / display_width))
                    orig_click_y = int(coords["y"] * (orig_h / display_height))
                    
                    st.session_state.clicks.append((orig_click_x, orig_click_y))
                    if len(st.session_state.clicks) > 2:
                        st.session_state.clicks = [st.session_state.clicks[-1]]
                    st.rerun()
                    
            # Clear buttons
            c_btn1, c_btn2 = st.columns(2)
            with c_btn1:
                if st.button("🧹 Clear Selection", use_container_width=True):
                    clear_selection()
                    st.rerun()
            with c_btn2:
                if st.button("🗑️ Clear All Fields", use_container_width=True, type="secondary"):
                    clear_all_fields()
                    st.rerun()
                    
        with col_fields:
            st.subheader("2. Define & Add Field")
            
            field_label = st.text_input(
                "Field Label Name",
                placeholder="e.g. Invoice_Number, Total_Amount, Date",
                disabled=len(st.session_state.clicks) < 2
            )
            
            if st.button("➕ Add Field", type="primary", use_container_width=True, disabled=len(st.session_state.clicks) < 2):
                if not field_label.strip():
                    st.error("Please enter a field label.")
                elif any(f["label"] == field_label.strip() for f in st.session_state.fields):
                    st.error(f"Field with label '{field_label.strip()}' already exists.")
                else:
                    x1, y1 = st.session_state.clicks[0]
                    x2, y2 = st.session_state.clicks[1]
                    
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x1 - x2)
                    h = abs(y1 - y2)
                    
                    st.session_state.fields.append({
                        "label": field_label.strip(),
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h)
                    })
                    st.session_state.saved_template = None
                    clear_selection()
                    st.success(f"Added field '{field_label.strip()}' successfully.")
                    st.rerun()
            
            st.subheader("3. Current Template Fields")
            if st.session_state.fields:
                df_data = [[f['label'], f['x'], f['y'], f['w'], f['h']] for f in st.session_state.fields]
                df = pd.DataFrame(df_data, columns=["Label", "X", "Y", "Width", "Height"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No template fields defined yet. Draw bounding boxes on the reference image.")
                
            st.subheader("4. Save Template Config")
            template_filename = st.text_input("Template Filename", value="invoice_template.json")
            
            if st.button("💾 Save Template", use_container_width=True, type="primary"):
                if not st.session_state.fields:
                    st.error("❌ Please define at least one field before saving.")
                elif not template_filename.strip():
                    st.error("❌ Please enter a valid filename.")
                else:
                    # Convert ref image to PNG bytes, then base64 string
                    _, buffer = cv2.imencode('.png', cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
                    ref_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    template_data = {
                        "width": orig_w,
                        "height": orig_h,
                        "reference_image_b64": ref_b64,
                        "fields": st.session_state.fields
                    }
                    
                    os.makedirs("templates", exist_ok=True)
                    filename_clean = template_filename.strip()
                    if not filename_clean.endswith(".json"):
                        filename_clean += ".json"
                        
                    save_path = os.path.join("templates", filename_clean)
                    try:
                        save_template(save_path, template_data)
                        st.success(f"✅ Saved template configuration to `{save_path}`!")
                        
                        # Store in session state for downloading and displaying
                        json_str = json.dumps(template_data, indent=4)
                        st.session_state.saved_template = {
                            "content": json_str.encode('utf-8'), # binary encoding for robust downloads
                            "filename": filename_clean,
                            "raw_json": json_str
                        }
                        # Store in session state for active template usage in Tab 2
                        st.session_state.active_template = {
                            "name": filename_clean,
                            "data": template_data
                        }
                    except Exception as e:
                        st.error(f"❌ Error saving template: {str(e)}")
            
            # Render download actions and view copy code persistently
            if st.session_state.saved_template is not None:
                # 1. Use this template directly action
                if st.button("⚡ Use this template for OCR Extraction", use_container_width=True, type="primary"):
                    st.success("👉 Template set as active! Now click the **'📄 Extract OCR'** tab at the top of the page to upload and process your skewed documents.")
                
                # 2. Native Streamlit download button using binary bytes
                st.download_button(
                    label="📥 Download Template JSON",
                    data=st.session_state.saved_template["content"],
                    file_name=st.session_state.saved_template["filename"],
                    mime="application/json",
                    use_container_width=True
                )
                
                # 2. Local deployment helper note
                st.info("💡 Note: Running locally? The template is already saved in your project's `templates/` folder.")
                
                # 3. Clipboard View/Copy Box fallback
                with st.expander("📋 View Template JSON (Click top-right 'Copy' if download fails)"):
                    st.code(st.session_state.saved_template["raw_json"], language="json")
    else:
        st.warning("⚠️ No reference image loaded. Please upload a reference image above to get started.")

# ----------------- TAB 2: EXTRACT OCR -----------------
with tab2:
    st.header("Document Processing & OCR")
    st.write("Upload a template config and a distorted/skewed document image to align and extract text.")
    
    col_inputs, col_results = st.columns([1, 1])
    
    with col_inputs:
        st.subheader("1. Upload Template & Skewed Document")
        
        # Check if an active template was just created in Tab 1
        use_active_tpl = False
        if st.session_state.active_template is not None:
            use_active_tpl = st.checkbox(
                f"🔄 Use active template '{st.session_state.active_template['name']}' (created in Tab 1)",
                value=True
            )
            
        uploaded_template = st.file_uploader(
            "Upload Template JSON",
            type=["json"],
            key="template_uploader",
            disabled=use_active_tpl
        )
        
        uploaded_doc = st.file_uploader("Upload Skewed Document Image", type=["png", "jpg", "jpeg"], key="doc_uploader")
        
        # Resolve skewed document image
        doc_img = None
        if uploaded_doc is not None:
            file_bytes = np.asarray(bytearray(uploaded_doc.read()), dtype=np.uint8)
            doc_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            doc_img = cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to test_skewed.png if it exists
            default_skewed_path = "test_skewed.png"
            if os.path.exists(default_skewed_path):
                doc_img = cv2.imread(default_skewed_path)
                doc_img = cv2.cvtColor(doc_img, cv2.COLOR_BGR2RGB)
                st.info("💡 Using default skewed test image (`test_skewed.png`).")
                
        btn_extract = st.button("⚡ Align & Extract Data", type="primary", use_container_width=True)

    with col_results:
        st.subheader("2. Alignment & OCR Results")
        
        if btn_extract:
            if doc_img is None:
                st.error("❌ Please upload or provide a skewed document image first.")
            else:
                template_data = None
                
                # Retrieve template configuration
                if use_active_tpl and st.session_state.active_template is not None:
                    template_data = st.session_state.active_template["data"]
                    st.info(f"💡 Using active template '{st.session_state.active_template['name']}' directly from Tab 1.")
                elif uploaded_template is not None:
                    try:
                        template_data = json.load(uploaded_template)
                    except Exception as e:
                        st.error(f"❌ Error parsing template JSON: {str(e)}")
                else:
                    # Fallback to templates/KTP.json
                    default_tpl_path = os.path.join("templates", "KTP.json")
                    if os.path.exists(default_tpl_path):
                        try:
                            with open(default_tpl_path, 'r', encoding='utf-8') as f:
                                template_data = json.load(f)
                            st.info("💡 Using default template (`templates/KTP.json`).")
                        except Exception as e:
                            st.error(f"❌ Error loading default template: {str(e)}")
                    else:
                        st.error("❌ Please upload a template JSON file or build one in Tab 1.")
                
                if template_data is not None:
                    ref_b64 = template_data.get("reference_image_b64")
                    if not ref_b64:
                        st.error("❌ The template does not contain a valid reference image.")
                    else:
                        try:
                            # Decode base64 reference image
                            ref_bytes = base64.b64decode(ref_b64)
                            np_arr = np.frombuffer(ref_bytes, dtype=np.uint8)
                            ref_img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            ref_img = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2RGB)
                            
                            # Perspective Alignment
                            with st.spinner("🔄 Aligning skewed document perspective..."):
                                aligned_img = align_images(ref_img, doc_img)
                            
                            st.success("✅ Perspective alignment successful!")
                            st.image(aligned_img, caption="Aligned Document Output", use_container_width=True)
                            
                            # OCR Field Extraction
                            with st.spinner("🤖 Extracting fields text via EasyOCR..."):
                                extractor = get_ocr_extractor()
                                extracted_data = extractor.extract_fields(aligned_img, template_data)
                            
                            # Render results table
                            st.subheader("📋 Extracted Results")
                            df_extracted = pd.DataFrame(
                                [[label, val] for label, val in extracted_data.items()],
                                columns=["Field Label", "Extracted Value"]
                            )
                            st.dataframe(df_extracted, use_container_width=True, hide_index=True)
                            
                        except AlignmentError as ae:
                            st.error(f"⚠️ Alignment Failed: {str(ae)}")
                        except Exception as e:
                            st.error(f"❌ Processing Error: {str(e)}")

# Footer
st.markdown("<div class='footer-container'>Configuration-Driven OCR Engine v1.0.0 • Developed with OpenCV & EasyOCR</div>", unsafe_allow_html=True)
