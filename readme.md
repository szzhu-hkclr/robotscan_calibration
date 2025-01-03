## Instruction
- The code is used for calibrating $^{tracker}T_{Link3}$ and DH parameters of last-three links based on the tracker data and robot kinematics.
- We denote each set of fixed first-three joints as a **group**. Each **group** relates to a $^{tracker}T_{Link3}$.
- In our code, we use ```visual_kinematics``` to compute the forward-kinematics, please adjust the DH parameter in the code if you want to adapt the code to your own robot.


### Dependencies (python = 3.8)

Before running the code, use ```pip``` to install the following libraries.
- pandas
```
pip install pandas
```

- open3d
- torch
```
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```
pip3 --no-cache-dir install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
python -m torch.utils.collect_env
python -c "import torch; print(torch.cuda.is_available())"
```

- opencv-python

- visual_kinematics
```
pip3 install visual-kinematics
```

- RoMa
```
pip3 install roma
```

### Usage

- Calibrate the $^{tracker}T_{Link3}$ of each group and DH parameters of the last-three links:
  1. For each group, capture at least 15 samples. Each sample contains the marker positions and the joint states. The marker should be linked to the end-effector.

     Note that at each group, the first-three joint states should be the same and the last-three joint states should span the work space as much as possible.
  
  2. First, run ```calib_aT3_init.py``` and you can get the initial solutions, i.e, ```aT3s_init.npy``` and ```p_init.npy```;
  
  3. Second, run ```calib_aT3_opt.py ``` and you can get the final solutions, i.e., ```aT3s_init.npy```.

     Note that in ```calib_aT3_opt.py ```, we also calib the DH parameter of last-three links, i.e., Lines 101-103. You should adjust the data according to your robot.

- Hand-eye calibration
  1. Capture 24 images belongs to 6 calibrated groups for the chessboard (24 = 4*6, you can also adjust the number). 
  
  2. First, run ```handeye_6DoF.py``` to get the camera intrinsics, distortion, and the $^{end}T_{cam}$ based on original 6-DoF kinematics and DH parameters.

  3. Second, based on the pre-computed intrinsics and distortion, run ```handeye_newkin.py``` to get new $^{end}T_{cam}$.

- Evaluate the accuracy
  1. For each pre-calibrated group, capture an image of the chessboard and the pointcloud (the unit should be mm).

  2. Adjust the related variables calibrated above in ```6DoF_vs_newkin.py``` and run it for evaluation.
     
     For example, you need to adjust the ```intrinsic_matrix```, ```dist```, ```dh_params```, ```end_T_cam_ori```, ```dh_params_new```, and ```end_T_cam```.


### Code Structure
- Calibrate the $^{tracker}T_{Link3}$ and DH parameters of the last-three links:
```
.
├── calib_aT3_init.py
├── calib_aT3_opt.py
├── calib_aT3_data
    ├── data_leika       
        ├── group1.csv       ### each line records the position of marker (mm)
        ├── group2.csv
        ├── ...
    ├── data_robot
        ├── group1.txt       ### each line records the joint states
        ├── group2.txt   
        ├── ...

``` 

- Hand-eye calibration based on $^{tracker}T_{Link3}$ of each group and new DH parameters:
```
.
├── handeye_6DoF.py
├── handeye_newkin.py
├── handeye_data
    ├── data_images       
        ├── group1          
           ├── $time1$_2DImage.png
           ├── $time2$_2DImage.png 
           ├── ...
        ├── group2
        ├── ...
    ├── data_robot
        ├── group1.txt       ### each line records the joint states
        ├── group2.txt   
        ├── ...

``` 

- Evaluate the accuracy, download the test_data in this [Link](https://hkclr-my.sharepoint.com/:u:/g/personal/tyhuang_hkclr_hk/EfGZLp1JI8ZGndlROqBqGj8BGKlApr0idKCS-i7QTSZ0CQ).
```
.
├── 6DoF_vs_newkin.py
├── test_data
    ├── image_data       
        ├── $time1$_2DImage.png
        ├── $time1$_textured.ply
        ├── $time2$_2DImage.png
        ├── $time2$_textured.ply
        ├── ...
    ├── robot_data.txt       ### each line records the joint states
```
