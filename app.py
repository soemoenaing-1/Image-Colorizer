import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from colorizers import eccv16, siggraph17
from colorizers.util import load_img, preprocess_img, postprocess_tens
from skimage import color
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Image Colorizer", layout="wide")
st.title("🖼️ AI Image Colorizer")

def resize_img(img, HW, resample=3):
    """Resize image to target dimensions"""
    return np.asarray(Image.fromarray(img).resize((HW[1], HW[0]), resample=resample))

def preprocess_img_enhanced(img_rgb_orig, HW=(384, 384), resample=3):
    """Enhanced preprocessing with custom resolution"""
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    
    img_lab_orig = color.rgb2lab(img_rgb_orig)
    img_lab_rs = color.rgb2lab(img_rgb_rs)

    img_l_orig = img_lab_orig[:,:,0]
    img_l_rs = img_lab_rs[:,:,0]

    tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
    tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

    return (tens_orig_l, tens_rs_l)

def postprocess_tens_enhanced(tens_orig_l, out_ab):
    """Enhanced postprocessing with upscaling"""
    HW_orig = tens_orig_l.shape[2:]
    HW = out_ab.shape[2:]

    if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
        out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
    else:
        out_ab_orig = out_ab

    out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
    return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))

def enhance_colors(img_array, saturation=1.3, contrast=1.15, sharpness=1.2):
    """Apply color enhancement to improve vibrancy and quality"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation)
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)
    
    return np.array(img).astype(np.float32) / 255.0

def apply_color_variation(img_array, variation):
    """Apply different color variations to the image"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    if variation == "original":
        # No enhancement - original colorization
        return img_array
    
    elif variation == "vibrant":
        # High saturation and contrast - vivid colors
        return enhance_colors(img_array, saturation=1.5, contrast=1.2, sharpness=1.3)
    
    elif variation == "warm":
        # Warm tones - more red/yellow
        img_array = enhance_colors(img_array, saturation=1.3, contrast=1.1, sharpness=1.15)
        # Apply warm filter by shifting colors
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        from PIL import ImageOps
        # Convert to warm by adjusting color balance
        img_arr = np.array(img).astype(np.float32)
        img_arr[:,:,0] = np.clip(img_arr[:,:,0] * 1.1, 0, 255)  # Red
        img_arr[:,:,1] = np.clip(img_arr[:,:,1] * 1.05, 0, 255)  # Green
        img_arr[:,:,2] = np.clip(img_arr[:,:,2] * 0.9, 0, 255)  # Blue
        return (img_arr / 255.0).astype(np.float32)
    
    elif variation == "cool":
        # Cool tones - more blue
        img_array = enhance_colors(img_array, saturation=1.2, contrast=1.1, sharpness=1.15)
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        # Convert to cool by adjusting color balance
        img_arr = np.array(img).astype(np.float32)
        img_arr[:,:,0] = np.clip(img_arr[:,:,0] * 0.9, 0, 255)  # Red
        img_arr[:,:,1] = np.clip(img_arr[:,:,1] * 1.0, 0, 255)  # Green
        img_arr[:,:,2] = np.clip(img_arr[:,:,2] * 1.15, 0, 255)  # Blue
        return (img_arr / 255.0).astype(np.float32)
    
    elif variation == "vintage":
        # Muted/vintage look - slightly faded
        img_array = enhance_colors(img_array, saturation=0.85, contrast=0.95, sharpness=1.0)
        return img_array
    
    elif variation == "cinematic":
        # Cinematic look - high contrast, desaturated
        return enhance_colors(img_array, saturation=0.9, contrast=1.25, sharpness=1.2)
    
    return img_array

# Sidebar for settings
st.sidebar.header("⚙️ Quality Settings")

# Resolution setting
resolution = st.sidebar.selectbox(
    "Processing Resolution",
    options=[256, 384, 512],
    index=1,
    help="Higher resolution = better quality but slower processing"
)

# Number of variations
num_variations = st.sidebar.selectbox(
    "Number of Color Variations",
    options=[1, 2, 3, 4],
    index=3,
    help="Generate multiple color versions"
)

# GPU option
use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=torch.cuda.is_available())

# Sample image option
st.sidebar.markdown("---")
st.sidebar.subheader("Sample Image")
use_sample = st.sidebar.checkbox("Use sample image", value=False, 
    help="Use built-in sample image for testing")
sample_image_path = "imgs/ansel_adams3.jpg"

# Upload image
if use_sample:
    uploaded_file = sample_image_path
else:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    if use_sample:
        img = load_img(uploaded_file)
    else:
        img = np.array(Image.open(uploaded_file).convert("RGB"))
    
    # Display original
    st.image(img, caption="Original Image", use_column_width=True)
    
    # Get original dimensions
    orig_h, orig_w = img.shape[:2]
    st.info(f"📐 Original size: {orig_w} x {orig_h}")
    
    # Load colorization model
    colorizer = siggraph17(pretrained=True).eval()
    
    if use_gpu and torch.cuda.is_available():
        colorizer.cuda()
        st.success("🚀 Using GPU acceleration")
    
    # Preprocess with enhanced resolution
    tens_l_orig, tens_l_rs = preprocess_img_enhanced(img, HW=(resolution, resolution))
    
    if use_gpu and torch.cuda.is_available():
        tens_l_rs = tens_l_rs.cuda()
    
    # Grayscale image
    img_bw = postprocess_tens_enhanced(tens_l_orig, torch.cat((0*tens_l_orig, 0*tens_l_orig), dim=1))
    
    # Display grayscale
    st.subheader("Grayscale")
    st.image(img_bw, use_column_width=True)
    
    # Define color variations based on selection
    variation_names = {
        1: ["vibrant"],
        2: ["original", "vibrant"],
        3: ["original", "vibrant", "warm"],
        4: ["vibrant", "warm", "cool", "vintage"]
    }
    
    variations = variation_names.get(num_variations, ["vibrant"])
    
    # Process and display all variations
    st.subheader("🎨 Colorized Results")
    
    results = []
    for i, variation in enumerate(variations):
        with st.spinner(f"Generating {variation} version..."):
            # Get base colorization
            out_colorized = postprocess_tens_enhanced(tens_l_orig, colorizer(tens_l_rs).cpu())
            
            # Apply variation
            out_colorized = apply_color_variation(out_colorized, variation)
            
            # Display with styled caption
            variation_labels = {
                "original": "Original",
                "vibrant": "Vibrant Colors",
                "warm": "Warm Tones",
                "cool": "Cool Tones",
                "vintage": "Vintage",
                "cinematic": "Cinematic"
            }
            
            st.image(out_colorized, caption=f"Variation {i+1}: {variation_labels.get(variation, variation)}", use_column_width=True)
            results.append((variation_labels.get(variation, variation), out_colorized))
    
    # Download options
    st.markdown("### 💾 Download Results")
    
    from io import BytesIO
    
    col1, col2, col3, col4 = st.columns(4)
    
    for idx, (name, img_result) in enumerate(results):
        buf = BytesIO()
        Image.fromarray((img_result * 255).astype(np.uint8)).save(buf, format="PNG")
        
        if idx == 0:
            col1.download_button(
                label=f"Download {name}",
                data=buf.getvalue(),
                file_name=f"colorized_{name.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
        elif idx == 1:
            col2.download_button(
                label=f"Download {name}",
                data=buf.getvalue(),
                file_name=f"colorized_{name.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
        elif idx == 2:
            col3.download_button(
                label=f"Download {name}",
                data=buf.getvalue(),
                file_name=f"colorized_{name.lower().replace(' ', '_')}.png",
                mime="image/png"
            )
        elif idx == 3:
            col4.download_button(
                label=f"Download {name}",
                data=buf.getvalue(),
                file_name=f"colorized_{name.lower().replace(' ', '_')}.png",
                mime="image/png"
            )

# Tips section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("""
- **4 Variations**: Get Vibrant, Warm, Cool, Vintage
- **Higher Resolution**: Use 512 for more details
- **Use Sample**: Test with built-in image
""")
