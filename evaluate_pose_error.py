import numpy as np
from scipy.spatial.transform import Rotation as R
from visual_kinematics.RobotSerial import RobotSerial

# Define DH parameters for the robot
dh_params = np.array([
    [0.55, 0.17, np.pi / 2, 0.0],
    [0.0, 0.88, 0.0, np.pi / 2],
    [0.0, 0.19, np.pi / 2, 0.0],
    [0.81, 0.0, -np.pi / 2, 0.0],
    [0.0, 0.0, np.pi / 2, 0.0],
    [0.115, 0.0, 0.0, 0.0]
])

# Create a robot object using the DH parameters
robot = RobotSerial(dh_params)

# Function to parse the data file
def parse_data(file_path):
    joint_states = []
    translations = []
    rotations = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 4):  # Process every 4 lines as a group
            # Handle blank or improperly formatted lines
            if i + 2 >= len(lines):
                break  # Skip if there are not enough lines left for a full group
            
            # Parse joint states
            joint_line = lines[i].strip()
            if not joint_line:  # Skip blank lines
                continue
            try:
                joint_values = [float(x) for x in joint_line.split(", ") if x.strip()]
                joint_states.append(joint_values)
            except ValueError:
                print(f"Skipping invalid joint state line: {joint_line}")
                continue
            
            # Parse translation
            translation_line = lines[i + 1].strip()
            try:
                translation_values = [float(x) for x in translation_line.split(": ")[1].strip("[]").split(", ") if x.strip()]
                translations.append(translation_values)
            except (IndexError, ValueError):
                print(f"Skipping invalid translation line: {translation_line}")
                continue
            
            # Parse rotation
            rotation_line = lines[i + 2].strip()
            try:
                rotation_values = [float(x) for x in rotation_line.split(": ")[1].strip("[]").split(", ") if x.strip()]
                rotations.append(rotation_values)
            except (IndexError, ValueError):
                print(f"Skipping invalid rotation line: {rotation_line}")
                continue
    
    return joint_states, translations, rotations

# Function to compute forward kinematics and compare with data
def evaluate_poses(file_path):
    """
    Evaluate pose errors and compute Euclidean distances in meters.
    
    Args:
        file_path (str): Path to the data file.
    
    Returns:
        List[float]: List of pose errors (Euclidean distances in meters).
    """
    joint_states, translations, rotations = parse_data(file_path)
    pose_errors = []
    translation_errors = []
    angular_differences = []
    
    for i, joint_state in enumerate(joint_states):
        # Compute forward kinematics
        try:
            robot.forward(np.array(joint_state))  # Compute all transformations
            Ts = robot.ts  # List of transformations from Link 1 to each link
            
            # Compute the transformation from Link 1 to Link 6
            T_end = np.eye(4)  # Identity matrix to start the multiplication
            for T in Ts:  # Multiply transformations from Link 1 to Link 6
                T_end = np.dot(T_end, T.t_4_4)
        except Exception as e:
            print(f"Error computing forward kinematics for joint state {i + 1}: {e}")
            continue

        # Extract translation and rotation (quaternion) from the transformation
        calculated_translation = T_end[:3, 3]
        calculated_rotation_matrix = T_end[:3, :3]
        calculated_quaternion = R.from_matrix(calculated_rotation_matrix).as_quat()
        
        # Get the recorded translation and rotation from the data file
        data_translation = np.array(translations[i])
        data_quaternion = np.array(rotations[i])
        
        # Compute translation error (Euclidean distance)
        translation_error = np.linalg.norm(calculated_translation - data_translation)
        translation_errors.append(translation_error)
        
        # Compute rotation error as angular difference
        # Ensure quaternions are normalized and account for double-cover property
        if np.dot(calculated_quaternion, data_quaternion) < 0:
            calculated_quaternion *= -1  # Flip the sign to ensure proper comparison
        angular_difference = 2 * np.arccos(np.clip(np.dot(calculated_quaternion, data_quaternion), -1.0, 1.0))
        angular_differences.append(angular_difference)
        
        pose_error = np.sqrt(translation_error**2)
        pose_errors.append(pose_error)
        
        # Print results
        print(f"Sample {i + 1}:")
        print(f"  Calculated Translation: {calculated_translation}")
        print(f"  Data Translation: {data_translation}")
        print(f"  Translation Error: {translation_error}")
        print(f"  Angular Difference (radians): {angular_difference}")
        print("")
    
    # Print statistics for translation and angular errors
    print("Translation Error Statistics:")
    print(f"  Minimum Translation Error: {np.min(translation_errors)} meters")
    print(f"  Maximum Translation Error: {np.max(translation_errors)} meters")
    print(f"  Mean Translation Error: {np.mean(translation_errors)} meters")
    print(f"  Median Translation Error: {np.median(translation_errors)} meters")
    print("")
    print("Angular Error Statistics:")
    print(f"  Minimum Angular Difference: {np.min(angular_differences)} radians")
    print(f"  Maximum Angular Difference: {np.max(angular_differences)} radians")
    print(f"  Mean Angular Difference: {np.mean(angular_differences)} radians")
    print(f"  Median Angular Difference: {np.median(angular_differences)} radians")
    print("")
    
    return pose_errors

# File path to the data file
file_path = "./kinematics_data/data_nachi.txt"

# Evaluate poses and calculate errors
errors = evaluate_poses(file_path)
print("All translation errors (Euclidean distances in meters):", errors)