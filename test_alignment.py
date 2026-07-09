import unittest
import numpy as np
import cv2
from alignment import align_images, AlignmentError

class TestAlignment(unittest.TestCase):
    def setUp(self):
        # Create a synthetic reference image (e.g., 600x800, 3 channels)
        self.height, self.width = 600, 800
        self.ref_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        # Draw distinctive visual features (text, rectangles, grid) so ORB can detect keypoints
        # Border
        cv2.rectangle(self.ref_img, (20, 20), (self.width - 20, self.height - 20), (0, 0, 0), 3)
        # Some inner rectangles representing boxes
        cv2.rectangle(self.ref_img, (100, 100), (300, 200), (50, 50, 50), -1)
        cv2.rectangle(self.ref_img, (400, 150), (700, 250), (100, 100, 100), 5)
        # Distinct lines
        cv2.line(self.ref_img, (50, 300), (750, 300), (0, 0, 0), 4)
        cv2.line(self.ref_img, (400, 50), (400, 550), (0, 0, 0), 2)
        # Add some text
        cv2.putText(self.ref_img, "TEMPLATE INVOICE ENGINE", (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(self.ref_img, "INVOICE NUMBER: 12345", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(self.ref_img, "TOTAL DUE: $999.99", (100, 480), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    def test_perfect_alignment(self):
        """Tests alignment when input is already perfectly aligned."""
        aligned = align_images(self.ref_img, self.ref_img.copy())
        self.assertEqual(aligned.shape, self.ref_img.shape)
        # Since it's identical, it should be an exact match
        mse = np.mean((self.ref_img - aligned) ** 2)
        self.assertLess(mse, 1.0)

    def test_skewed_alignment(self):
        """Tests alignment when the input image is rotated and translated (warped)."""
        # Create a homography matrix for rotation + translation
        # Rotate by 4 degrees and translate by dx=20, dy=15
        angle = 4.0
        scale = 0.98
        center = (self.width / 2, self.height / 2)
        
        # Get rotation matrix (2x3)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        # Convert to 3x3 homography matrix
        h_matrix = np.eye(3)
        h_matrix[:2, :] = rot_mat
        # Add translation
        h_matrix[0, 2] += 20
        h_matrix[1, 2] += -15

        # Create the warped input image
        input_img = cv2.warpPerspective(self.ref_img, h_matrix, (self.width, self.height), borderValue=(255, 255, 255))

        # Run our alignment function
        try:
            aligned = align_images(self.ref_img, input_img)
        except AlignmentError as e:
            self.fail(f"Alignment failed unexpectedly: {e}")

        self.assertEqual(aligned.shape, self.ref_img.shape)

        # To evaluate alignment quality, compare the inner region (excluding edges where warping introduces border artifacts)
        margin = 50
        ref_crop = self.ref_img[margin:-margin, margin:-margin]
        aligned_crop = aligned[margin:-margin, margin:-margin]
        
        mse = np.mean((ref_crop - aligned_crop) ** 2)
        
        # A good alignment should reconstruct the image very closely
        # In typical ORB-Homography setups, MSE of the non-border area should be very low (< 50 or so)
        print(f"Computed MSE between reference and aligned image: {mse:.4f}")
        self.assertLess(mse, 80.0)

    def test_layout_mismatch_error(self):
        """Tests that a totally mismatched image raises AlignmentError."""
        # Create a completely blank image
        mismatched_img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        # Draw a single circle (not enough features for matching)
        cv2.circle(mismatched_img, (300, 300), 50, (0, 0, 0), -1)

        with self.assertRaises(AlignmentError):
            align_images(self.ref_img, mismatched_img)

if __name__ == '__main__':
    unittest.main()
