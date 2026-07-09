import cv2
import numpy as np

def generate_images():
    # 1. Create reference image
    height, width = 600, 800
    ref_img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Draw border
    cv2.rectangle(ref_img, (15, 15), (width - 15, height - 15), (50, 50, 50), 3)
    
    # Header
    cv2.putText(ref_img, "OFFICIAL INVOICE", (220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Invoice details
    cv2.putText(ref_img, "Invoice No: INV-2026-999", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(ref_img, "Date: July 07, 2026", (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Item Table Header
    cv2.rectangle(ref_img, (80, 320), (720, 370), (220, 220, 220), -1)
    cv2.putText(ref_img, "Description", (100, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(ref_img, "Amount", (550, 355), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Item Row
    cv2.putText(ref_img, "AI Consulting Services (10 hrs)", (100, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(ref_img, "$1,500.00", (550, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Total Due box
    cv2.rectangle(ref_img, (500, 480), (720, 530), (240, 240, 240), -1)
    cv2.putText(ref_img, "TOTAL: $1,500.00", (510, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Save reference
    cv2.imwrite("test_ref.png", ref_img)
    print("Saved test_ref.png")
    
    # 2. Create warped (skewed) input image
    # Rotate by 3 degrees, scale by 0.97, and shift dx=15, dy=10
    angle = 3.0
    scale = 0.97
    center = (width / 2, height / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
    
    h_matrix = np.eye(3)
    h_matrix[:2, :] = rot_mat
    h_matrix[0, 2] += 15
    h_matrix[1, 2] += 10
    
    # Add minor lighting variation (slightly darker and shadow gradient)
    skewed_img = cv2.warpPerspective(ref_img, h_matrix, (width, height), borderValue=(255, 255, 255))
    
    # Simulate camera illumination change
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    # Simple linear illumination gradient (top-left is brighter, bottom-right is darker)
    illumination = (x_coords + y_coords) / (width + height) * 30  # Max decrease of 30 pixel values
    illumination = np.repeat(illumination[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    
    skewed_img = cv2.subtract(skewed_img, illumination)
    
    # Save skewed
    cv2.imwrite("test_skewed.png", skewed_img)
    print("Saved test_skewed.png")

if __name__ == "__main__":
    generate_images()
