##  Problem Statement – Camera Radial Distortion Estimation

In this project, the goal is to estimate **camera radial distortion parameters** from a single image of a **planar rectangular grid** (such as a checkerboard or tiled floor) captured using an **unknown camera** with **unknown distortion**.

The image may include:
- Partial occlusion of the grid  
- Perspective tilt or oblique viewing angles  
- Moderate lighting variations and noise  

###  Objectives
1. Formulate a **robust cost function** to estimate distortion parameters, along with optional camera intrinsics, extrinsics, and principal point.  
2. Build a **robust optimization pipeline** to refine the estimated parameters.  
3. Use **RANSAC** to detect and remove outliers.  
4. **Undistort** the image using the estimated parameters and reconstruct an accurate grid on the undistorted plane.  
5. **Reproject** the undistorted grid back into the original distorted image and compute **residuals** and **reprojection error** for validation.

This project demonstrates practical understanding of **camera calibration**, **radial distortion modeling**, and **geometric optimization** in computer vision.


## References  
To apply the concept of radial-distortion calibration and estimation, we referred to the following key research works and incorporated their methodologies into our implementation:

1. **Lens distortion correction based on one chessboard pattern image**  
   *Authors: Yubin Wu, Shixiong Jiang, Zhenkun Xu, Song Zhu, Danhua Cao.*  
   (Frontiers of Optoelectronics, 2015) :contentReference[oaicite:0]{index=0}  
   This paper inspired our checkerboard-based calibration approach.  
   **In our implementation:**  
   - We use detection of internal corners on a checkerboard pattern (`detect_corners_robust`).  
   - We run a non-linear optimisation (`refine_parameters`) to iteratively minimise distortion residuals — following the iterative scheme described in the paper.

2. **Automatic Radial Distortion Estimation from a Single Image**  
   *Authors: Faisal Bukhari & Matthew N. Dailey.*  
   (Journal of Mathematical Imaging and Vision, 2013) :contentReference[oaicite:1]{index=1}  
   This work guided our usage of single-image radial distortion estimation (without multiple views).  
   **In our implementation:**  
   - We adopt the distortion model `x_d = x_u (1 + k1 r² + k2 r⁴)`, aligning with the models discussed in the paper.  
   - We incorporate `apply_ransac` for robust inlier filtering of correspondences — analogous to their robust arc / line estimation method.

3. **Robust radial-distortion estimation using good circular arcs**  
   *Authors: X. Zhang et al.*  
   (ScienceDirect / Pattern Recognition Letters, 2015) :contentReference[oaicite:2]{index=2}  
   This paper emphasises using line/arc constraints for radial distortion correction.  
   **In our implementation:**  
   - Although our pipeline uses a checkerboard pattern rather than arbitrary arcs, the underlying notion of using geometric structure (straight lines → projected curves) to guide optimisation is adopted in our error-minimisation step (`compute_reprojection_error` + `refine_parameters`).  

---

These works together guided the design choices for our calibration tool — helping us build a pipeline that handles single-image distortion estimation, outlier removal, parameter refinement and undistortion.

<div align="center">
  <table><tr>
    <td align="center">
      **Input Image**<br>
      <img src="/input_images/image-005.jpg" alt="Input Image" width="250">
    </td>
    <td align="center">
      **Output Image**<br>
      <img src="/output_images/4_combined_results.jpg" alt="Output Image" width="250">
    </td>
  </tr></table>
</div>

# Computer_Vision Project

---

##  Getting Started

##### Follow these steps to clone the repository, set up the environment, and run the project.

---

### 🚀 Setup Instructions

#### 1. Clone the Repository
You can clone this repository using the following command:

```bash
git clone https://github.com/Savale-pandurang/Computer_Vision.git
```
#### 2. Navigate to the Project Directory

Change your current directory to the newly cloned project folder:
```
cd Computer_Vision
```
#### 3. Install required dependencies
```
pip install -r requirements.txt
```
#### 4. Run the project
```
python3 main.py
```
#### give input_path 
```
input_images/image-005.jpg
