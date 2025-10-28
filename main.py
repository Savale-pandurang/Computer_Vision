import cv2
import numpy as np
from scipy.optimize import least_squares

"""
    Performs single-image radial distortion calibration.
    Distortion model: x_d = x_u(1 + k1*r^2 + k2*r^4)
    where r^2 = x_u^2 + y_u^2 (normalized by focal length)
    """
class RadialDistortionCalibrator:
    def __init__(self, image_path, grid_size=(10, 7)):
        """Initialize with image path and checkerboard grid size."""
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise RuntimeError(f"Cannot load image: {image_path}")
        
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray.shape
        self.grid_size = grid_size
        
    """Detect checkerboard corners using multiple strategies."""
    def detect_corners_robust(self):
        print("Detecting corners...")
        
        ret, corners = cv2.findChessboardCorners(
            self.gray, self.grid_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_NORMALIZE_IMAGE +
            cv2.CALIB_CB_FAST_CHECK
        )
        
        if not ret:
            print("  Standard detection failed, trying with preprocessing...")
            processed = cv2.GaussianBlur(self.gray, (5, 5), 0)
            processed = cv2.equalizeHist(processed)
            ret, corners = cv2.findChessboardCorners(
                processed, self.grid_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        
        if not ret:
            raise RuntimeError("Could not detect checkerboard corners")
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            self.gray, corners, (11, 11), (-1, -1), criteria
        )
        
        print(f"  Detected {len(corners_refined)} corners")
        return corners_refined.reshape(-1, 2)
    
    """Generate 3D object points for the planar checkerboard."""
    def create_object_points(self, square_size=1.0):
        objp = np.zeros((self.grid_size[0] * self.grid_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.grid_size[0], 0:self.grid_size[1]].T.reshape(-1, 2)
        objp *= square_size
        return objp
    
    """Compute initial camera parameters using OpenCV."""
    def initial_calibration(self, corners, objpoints):
        print("Initial calibration...")
        
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            [objpoints], [corners.reshape(-1, 1, 2)], 
            (self.w, self.h), None, None,
            flags=cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST
        )
        
        return K, dist, rvecs[0], tvecs[0]
    
    """Apply RANSAC to filter out outlier points."""
    def apply_ransac(self, corners, objpoints, K, dist, rvec, tvec, threshold=2.0, iterations=1000):
        print("Applying RANSAC...")
        
        n_points = len(corners)
        best_inliers = []
        best_count = 0
        
        for _ in range(iterations):
            sample_idx = np.random.choice(n_points, 8, replace=False)
            projected, _ = cv2.projectPoints(objpoints[sample_idx], rvec, tvec, K, dist)
            projected = projected.reshape(-1, 2)
            
            all_projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
            all_projected = all_projected.reshape(-1, 2)
            errors = np.linalg.norm(corners - all_projected, axis=1)
            inliers = np.where(errors < threshold)[0]
            
            if len(inliers) > best_count:
                best_count = len(inliers)
                best_inliers = inliers
        
        print(f"  Inliers: {best_count}/{n_points}")
        return best_inliers
    
    """Compute reprojection residuals for optimization."""
    def residual_function(self, params, objpoints, imgpoints, w, h):
        fx, fy, cx, cy, k1, k2 = params[:6]
        rvec = params[6:9]
        tvec = params[9:12]
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.array([k1, k2, 0, 0, 0])
        
        projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        residuals = (imgpoints - projected).ravel()
        return residuals
    
    """Refine distortion parameters through non-linear optimization."""
    def refine_parameters(self, corners, objpoints, K, dist, rvec, tvec, inliers):
        print("Refining parameters...")
        
        corners_inliers = corners[inliers]
        objpoints_inliers = objpoints[inliers]
        
        params0 = np.concatenate([
            [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
            dist.ravel()[:2],
            rvec.ravel(),
            tvec.ravel()
        ])
        
        result = least_squares(
            self.residual_function,
            params0,
            args=(objpoints_inliers, corners_inliers, self.w, self.h),
            method='trf',
            verbose=0
        )
        
        params = result.x
        K_refined = np.array([[params[0], 0, params[2]], 
                             [0, params[1], params[3]], 
                             [0, 0, 1]])
        dist_refined = np.array([params[4], params[5], 0, 0, 0])
        rvec_refined = params[6:9].reshape(3, 1)
        tvec_refined = params[9:12].reshape(3, 1)
        
        return K_refined, dist_refined, rvec_refined, tvec_refined
    
    """Generate the undistorted version of the input image."""
    def undistort_image(self, K, dist):
        print("Undistorting image...")
        
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (self.w, self.h), 1, (self.w, self.h))
        undistorted = cv2.undistort(self.img, K, dist, None, newK)
        return undistorted, newK
    
    """Compute the mean reprojection error."""
    def compute_reprojection_error(self, corners, objpoints, K, dist, rvec, tvec):
        projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(corners - projected, axis=1)
        mean_error = np.mean(errors)
        return mean_error, errors
    
    """Visualize detection, reprojection, and undistortion results."""
    def visualize_results(self, corners, objpoints, K, dist, rvec, tvec, undistorted, inliers):
        img_corners = self.img.copy()
        for i, corner in enumerate(corners):
            color = (0, 255, 0) if i in inliers else (0, 0, 255)
            cv2.circle(img_corners, tuple(corner.astype(int)), 5, color, -1)
        
        img_reproj = self.img.copy()
        projected, _ = cv2.projectPoints(objpoints, rvec, tvec, K, dist)
        projected = projected.reshape(-1, 2)
        
        for i in range(len(corners)):
            if i in inliers:
                cv2.circle(img_reproj, tuple(corners[i].astype(int)), 5, (0, 255, 0), 2)
                cv2.circle(img_reproj, tuple(projected[i].astype(int)), 3, (0, 0, 255), -1)
                cv2.line(img_reproj, tuple(corners[i].astype(int)), tuple(projected[i].astype(int)), (255, 0, 0), 1)
        
        h1, w1 = img_corners.shape[:2]
        max_width = 800
        scale = max_width / max(w1, w1)
        new_h, new_w = int(h1 * scale), int(w1 * scale)
        
        img_corners_resized = cv2.resize(img_corners, (new_w, new_h))
        img_reproj_resized = cv2.resize(img_reproj, (new_w, new_h))
        original_resized = cv2.resize(self.img, (new_w, new_h))
        undistorted_resized = cv2.resize(undistorted, (new_w, new_h))
        
        top_row = np.hstack([original_resized, img_corners_resized])
        bottom_row = np.hstack([img_reproj_resized, undistorted_resized])
        combined = np.vstack([top_row, bottom_row])
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Detected Corners', (new_w + 10, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Reprojection', (10, new_h + 30), font, 1, (0, 255, 0), 2)
        cv2.putText(combined, 'Undistorted', (new_w + 10, new_h + 30), font, 1, (0, 255, 0), 2)
        
        cv2.imwrite('1_detected_corners.jpg', img_corners)
        cv2.imwrite('2_reprojection.jpg', img_reproj)
        cv2.imwrite('3_undistorted.jpg', undistorted)
        cv2.imwrite('4_combined_results.jpg', combined)
        
        print("\nVisualization images saved:")
        print("  - 1_detected_corners.jpg")
        print("  - 2_reprojection.jpg")
        print("  - 3_undistorted.jpg")
        print("  - 4_combined_results.jpg")
    
    def calibrate(self):
        """Complete calibration pipeline."""
        print("="*60)
        print("ROBUST RADIAL DISTORTION CALIBRATION")
        print("="*60)
        
        corners = self.detect_corners_robust()
        objpoints = self.create_object_points()
        K, dist, rvec, tvec = self.initial_calibration(corners, objpoints)
        inliers = self.apply_ransac(corners, objpoints, K, dist, rvec, tvec)
        K_refined, dist_refined, rvec_refined, tvec_refined = self.refine_parameters(
            corners, objpoints, K, dist, rvec, tvec, inliers
        )
        undistorted, newK = self.undistort_image(K_refined, dist_refined)
        
        mean_error, errors = self.compute_reprojection_error(
            corners[inliers], objpoints[inliers], K_refined, dist_refined, rvec_refined, tvec_refined
        )
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print("\nCamera Intrinsic Matrix (K):")
        print(K_refined)
        print(f"\nFocal lengths: fx={K_refined[0,0]:.2f}, fy={K_refined[1,1]:.2f}")
        print(f"Principal point: cx={K_refined[0,2]:.2f}, cy={K_refined[1,2]:.2f}")
        print(f"\nDistortion Coefficients [k1, k2]:")
        print(f"k1 = {dist_refined[0]:.6f}")
        print(f"k2 = {dist_refined[1]:.6f}")
        print(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
        print(f"Max Reprojection Error: {np.max(errors):.4f} pixels")
        print(f"Min Reprojection Error: {np.min(errors):.4f} pixels")
        print("="*60)
        
        self.visualize_results(
            corners, objpoints, K_refined, dist_refined, rvec_refined, tvec_refined, undistorted, inliers
        )
        
        return {
            'K': K_refined,
            'dist': dist_refined,
            'rvec': rvec_refined,
            'tvec': tvec_refined,
            'undistorted': undistorted,
            'mean_error': mean_error,
            'inliers': inliers
        }


if __name__ == "__main__":
    calibrator = RadialDistortionCalibrator(
        image_path="/Computer_Vision/input_images/image-005.jpg",
        grid_size=(9, 6)
    )
    
    results = calibrator.calibrate()
    
    cv2.imwrite('undistorted_output.jpg', results['undistorted'])
    print("\n" + "="*60)
    print("OUTPUT FILES SAVED")
    print("="*60)
