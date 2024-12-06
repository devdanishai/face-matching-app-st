import streamlit as st
from deepface import DeepFace
from PIL import Image
import os
import cv2
import numpy as np

# Title of the app
st.title("👨‍🦱 Face Matching App 🔮")
st.title("🎯 DeepLearning Project 🎲")

# Use case points
st.subheader("Use Cases")
st.markdown("""
- **ID Verification**
- **Face Recognition**
- **Access Control**
- **Emotion Analysis**
- **Social Media Tagging**
- **Surveillance**
- **Attendance System**
""")

def is_valid_image(image_file):
    try:
        if image_file is None:
            return False
        img = Image.open(image_file)
        img.verify()  # Verify if it's a valid image
        return True
    except Exception:
        return False

def process_image(img):
    # Convert uploaded file to numpy array
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    return opencv_img

# Upload first image
st.subheader("Upload the 'Source' image")
img1 = st.file_uploader("Choose Source image...", type=["jpg", "jpeg", "png"])

# Upload second image
st.subheader("Upload 'Comparison' image")
img2 = st.file_uploader("Choose Comparison image...", type=["jpg", "jpeg", "png"])

# Button to check identity
if st.button("Check Identity"):
    if img1 is not None and img2 is not None:
        try:
            # Validate images
            if not is_valid_image(img1) or not is_valid_image(img2):
                st.error("Please upload valid image files.")
                st.stop()

            # Reset file pointers
            img1.seek(0)
            img2.seek(0)

            # Save the images to temporary location
            img1_path = "temp_img1.jpg"
            img2_path = "temp_img2.jpg"

            # Process and save images using OpenCV
            cv2.imwrite(img1_path, process_image(img1))
            cv2.imwrite(img2_path, process_image(img2))

            try:
                # Perform verification with additional parameters
                result = DeepFace.verify(
                    img1_path=img1_path,
                    img2_path=img2_path,
                    enforce_detection=False,
                    model_name="VGG-Face"
                )

                # Display the result
                if result["verified"]:
                    st.success("The images are of the same person!")
                    st.write(f"Confidence Score: {result.get('distance', 'N/A')}")
                else:
                    st.error("The images are NOT of the same person!")
                    st.write(f"Confidence Score: {result.get('distance', 'N/A')}")

                # Display images side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.image(img1, caption="First Image", use_container_width=True)

                with col2:
                    st.image(img2, caption="Second Image", use_container_width=True)

            except Exception as e:
                st.error(f"Error during face verification: {str(e)}")

            # Cleanup temporary files
            try:
                os.remove(img1_path)
                os.remove(img2_path)
            except:
                pass

        except Exception as e:
            st.error(f"Error processing images: {str(e)}")

    else:
        st.error("Please upload both images.")