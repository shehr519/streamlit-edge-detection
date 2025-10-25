import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Edge Detection Visualizer", layout="wide")
st.title("Edge Detection Visualizer")

# image handling
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sidebar 
    st.sidebar.header("🔍 Edge Detection Controls")
    algorithm = st.sidebar.radio("Select an Algorithm", ["Sobel", "Laplacian", "Canny"])

    # alogorithm parameters
    st.sidebar.subheader("⚙️ Adjust Parameters")
    if algorithm == "Canny":
        lower = st.sidebar.slider("Lower Threshold", 0, 255, 50)
        upper = st.sidebar.slider("Upper Threshold", 0, 255, 150)
        sigma = st.sidebar.slider("Sigma for Gaussian Blur", 0.0, 3.0, 1.0)
        ksize = st.sidebar.slider("Canny Kernel Size (odd only)", 1, 15, 3, step=2)
        apply_blur = st.sidebar.checkbox("Apply Gaussian Blur", value=True)
        if apply_blur:
          blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        else:
          blurred = gray
        edges = cv2.Canny(blurred, lower, upper)
        edges = cv2.convertScaleAbs(edges)


    elif algorithm == "Sobel":
        ksize = st.sidebar.slider("Sobel Kernel Size (odd only)", 1, 15, 3, step=2)
        direction = st.sidebar.radio("Gradient Direction", ["X", "Y", "Both"])
        
        if direction == "X":
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        elif direction == "Y":
            edges = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        else:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
            edges = cv2.magnitude(grad_x, grad_y)
        edges = cv2.convertScaleAbs(edges)

    elif algorithm == "Laplacian":
        ksize = st.sidebar.slider("Laplacian Kernel Size (odd only)", 1, 15, 3, step=2)
        edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        edges = cv2.convertScaleAbs(edges)
    st.markdown("---")
    realtime = st.checkbox("Realtime update", value=True)
    if not realtime:
        apply_btn = st.button("Apply")
    else:
        apply_btn = True
   

    # Display images side by side
    col1, col2, col3 = st.columns([1, 0.1, 1])
    with col1:
        st.subheader("**Input Image**")
        st.image(image,  use_container_width=True)
    with col3:
        st.subheader(f"**Output Image ({algorithm})**")
        st.image(edges,  use_container_width=True)

else:
    st.info("👆 Please upload an image to start.")

