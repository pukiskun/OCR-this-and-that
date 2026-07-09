import cv2
import numpy as np

class AlignmentError(Exception):
    """Exception raised when document alignment/registration fails."""
    pass

def align_images(ref_img: np.ndarray, input_img: np.ndarray, max_features: int = 3000, keep_percent: float = 0.15) -> np.ndarray:
    """
    Aligns input_img to the perspective of ref_img using ORB and Homography.
    Throws AlignmentError if alignment fails.
    """
    if ref_img is None or input_img is None:
        raise AlignmentError("Input images cannot be None.")

    # 1. Convert both reference and input images to grayscale.
    if len(ref_img.shape) == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img.copy()

    if len(input_img.shape) == 3:
        input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        input_gray = input_img.copy()

    # Apply cv2.adaptiveThreshold for illumination robustness.
    # Using Gaussian adaptive thresholding.
    ref_thresh = cv2.adaptiveThreshold(
        ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    input_thresh = cv2.adaptiveThreshold(
        input_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # 2. Detect features and compute descriptors using ORB.
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # We detect on the thresholded binary images as specified in the blueprint
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_thresh, None)
    keypoints_input, descriptors_input = orb.detectAndCompute(input_thresh, None)

    # Fallback to grayscale if thresholded images yield no keypoints (safety guard)
    if descriptors_ref is None or len(keypoints_ref) < 10:
        keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_gray, None)
    if descriptors_input is None or len(keypoints_input) < 10:
        keypoints_input, descriptors_input = orb.detectAndCompute(input_gray, None)

    if descriptors_ref is None or descriptors_input is None:
        raise AlignmentError("Document layout mismatch. Please select the correct template.")

    # 3. Match features using cv2.BFMatcher.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_ref, descriptors_input)

    # 4. Sort matches by distance and retain top N matches.
    matches = sorted(matches, key=lambda x: x.distance)

    # Check for minimum number of matches
    if len(matches) < 4:
        raise AlignmentError("Document layout mismatch. Please select the correct template.")

    num_keep = max(4, int(len(matches) * keep_percent))
    good_matches = matches[:num_keep]

    # Extract coordinates of matched keypoints
    pts_ref = np.zeros((len(good_matches), 2), dtype=np.float32)
    pts_input = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        pts_ref[i, :] = keypoints_ref[match.queryIdx].pt
        pts_input[i, :] = keypoints_input[match.trainIdx].pt

    # 5. Compute the transformation matrix using cv2.findHomography with cv2.RANSAC.
    h_matrix, mask = cv2.findHomography(pts_input, pts_ref, cv2.RANSAC, 5.0)

    if h_matrix is None:
        raise AlignmentError("Document layout mismatch. Please select the correct template.")

    # Verify that we have enough RANSAC inliers (at least 15 for robust matching)
    inliers = np.sum(mask) if mask is not None else 0
    if inliers < 15:
        raise AlignmentError("Document layout mismatch. Please select the correct template.")

    # 6. Warp the input image to match the exact dimension and alignment of the reference image.
    height, width = ref_img.shape[:2]
    aligned_img = cv2.warpPerspective(input_img, h_matrix, (width, height))

    return aligned_img
