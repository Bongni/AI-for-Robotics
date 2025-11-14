#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from ai4r_interfaces.msg import EscAndSteeringPercent
from ai4r_interfaces.msg import ConePointsArray
from ai4r_interfaces.msg import Imu
from ai4r_interfaces.msg import TofSensor
from std_msgs.msg import String, UInt16, Float32, Header
from rclpy.parameter import Parameter
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from geometry_msgs.msg import TransformStamped
import tf2_ros

from stable_baselines3 import PPO

import numpy as np

FSM_STATE_NOT_PUBLISHING_ACTIONS   = 1
FSM_STATE_PUBLISHING_ZERO_ACTIONS  = 2
FSM_STATE_PUBLISHING_POLICY_ACTION = 3

LIST_OF_FSM_STATES = [FSM_STATE_NOT_PUBLISHING_ACTIONS, FSM_STATE_PUBLISHING_ZERO_ACTIONS, FSM_STATE_PUBLISHING_POLICY_ACTION]

NUMBER_OF_ZERO_CONE_DETECTIONS_BEFORE_ZERO_ACTIONS = 5


RL_MODEL_PATH = "./models/model.zip"


# Create a class which inherits from the rclpy Node class (hence a superset of “rclpy.node.Node”).
class PolicyNode(Node):
    
    # CONSTRUCTOR FUNCTION
    def __init__(self):
        # Initialise the node object (passing the node name as an argument)
        super().__init__('policy_node')

        # Log the namespace
        self.get_logger().info("[POLICY NODE] starting __init__ with node namespace = " + self.get_namespace())

        self.rl_model = PPO.load(RL_MODEL_PATH)

        # Initialise the FSM state
        self.fsm_state = FSM_STATE_PUBLISHING_ZERO_ACTIONS

        # Initialise a counter for counting the number of zero cone detections in a row
        self.counts_of_zero_cone_detections_in_a_row_ = 0

        # Declare the parameters of this node
        self.declare_parameters(
            namespace='',
            parameters=[
                ("timer_period", rclpy.Parameter.Type.DOUBLE),
            ])
        self.timer_period = self.get_parameter_or("timer_period",Parameter("default_timer_period",Parameter.Type.DOUBLE,0.5)).value

        # Create ROS2 publishers
        # > For publishing the FSM state
        self.fsm_state_publisher_ = self.create_publisher(String, 'policy_fsm_state', 10)
        # > For publishing the policy actions
        self.policy_action_publisher_ = self.create_publisher(EscAndSteeringPercent, 'esc_and_steering_set_point_percent_action', 10)
        # > For publishing the IMU heading angle
        self.imu_heading_angle_publisher_ = self.create_publisher(Float32, 'imu_heading_angle', 10)
        # > For publishing debug1 messages
        self.debug1_publisher_ = self.create_publisher(Float32, 'debug1', 10)
        # > For publishing debug2 messages
        self.debug2_publisher_ = self.create_publisher(Float32, 'debug2', 10)
        # > For publishing the ToF point cloud
        self.tof_pointcloud_publisher_ = self.create_publisher(PointCloud2, "tof_point_cloud_vl53l5cx", 10)

        # TF broadcaster for world->car and car->ToF frames
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        # Create ROS2 subscribers
        # > For subscribing to the CV cone detection data
        self.cv_subscription = self.create_subscription(ConePointsArray, 'cone_points', self.cv_cone_detection_callback, 10)
        # > For subscribing to requests to transition the state
        self.fsm_transition_request_subscription = self.create_subscription(UInt16, 'policy_fsm_transition_request', self.fsm_transition_request_callback, 10)
        # > For subscribing to VL53L5CX ToF data
        self.tof_subscription = self.create_subscription(TofSensor, 'tof_sensor_vl53l5cx', self.tof_callback, 10)
        # > For subscribing to IMU data
        self.imu_subscription = self.create_subscription(Imu, 'imu_sensor', self.imu_callback, 10)
        # > Prevent unused variable warning
        self.cv_subscription
        self.fsm_transition_request_subscription
        self.tof_subscription
        self.imu_subscription

        # Initialize a class variable for the most recent heading angle measurement from the IMU
        self.heading_angle_from_imu = 0.0
        self.heading_angle_for_tare = 0.0

        # Initialize a class variable for the roll and pitch measurement from the IMU
        self.roll_angle_from_imu = 0.0
        self.pitch_angle_from_imu = 0.0

        # Initialize class variables for the most recent tof measurements
        self.tof_usec_since_last_msg = 0
        self.tof_silicon_temp_degc = 0.0
        self.tof_sensor_id = 0
        self.tof_distance_array_meters = np.array([], dtype=np.float32)
        self.tof_status_array = np.array([], dtype=np.uint8)

        # Initialize class variables related to transforming tof measurements to car frame
        self.tof_xyz_array_meters_in_tof_frame = np.array([], dtype=np.float32)
        self.tof_xyz_array_meters_in_car_frame = np.array([], dtype=np.float32)
        self.tof_translation_vector_meters_in_car_frame = np.array([0.0,0.0,0.25], dtype=np.float32)
        self.world_frame_id = "world"
        self.car_frame_id = "car_frame"
        self.tof_pointcloud_frame_id = "tof_frame"
        self.tof_should_use_roll_and_pitch_for_rotation_into_car_frame = True

        # + Initialize variables for PID control
        self.prev_lateral_error = 0.0  # lateral error at previous time step for D term
        self.integral_lateral_error = 0.0  # integral of lateral error for I term

        # Create a timer that is used for continually publishing the FSM state
        # > First argument is the duration between 2 callbacks (in seconds).
        # > Second argument is the callback function.
        self.create_timer(float(self.timer_period), self.timer_callback)
        # Timer for periodic TF publishing
        self.tf_timer = self.create_timer(0.05, self.publish_transforms)



    # CALLBACK FUNCTION: for the CV line detection subscription 
    def cv_cone_detection_callback(self, msg):
        # Log the data received for debugging purposes:
        # self.get_logger().info("[POLICY NODE] Cone detection points: \"%s\"" % msg.n)

        # Return if in the "not publishing actions" state
        if (self.fsm_state == FSM_STATE_NOT_PUBLISHING_ACTIONS):
            return

        # Default the actions to zero
        esc_action = 0.0
        steering_action = 0.0

        # Run the policy, if in the "publishing policy" state
        if (self.fsm_state == FSM_STATE_PUBLISHING_POLICY_ACTION):

            # Variables pre-declared for the purposes of debug publishing
            # > Note: only published if this number is changed by the policy code below
            distance_to_target_line = -9999

            # Extract the number of cones
            num_cones = msg.n

            # Extract the coordinates and colour of each cone
            # > NOTE: these are Python lists of length "num_cones"
            x_coords = msg.x
            y_coords = msg.y
            cone_colour = msg.c

            # Extract the current heading angle into a local vairable
            heading_angle_in_radians = self.heading_angle_from_imu - self.heading_angle_for_tare
            # > "Unwrap" the angle to be between [-pi,pi]
            if (heading_angle_in_radians > np.pi):
                heading_angle_in_radians -= 2.0*np.pi
            if (heading_angle_in_radians < -np.pi):
                heading_angle_in_radians += 2.0*np.pi

            # Extract the current roll and pitch angles into a local variable
            roll_angle_in_radians  = self.roll_angle_from_imu
            pitch_angle_in_radians = self.pitch_angle_from_imu

            # Extract the tof measurements into local variables
            tof_usec_since_last_msg = self.tof_usec_since_last_msg
            tof_silicon_temp_degc = self.tof_silicon_temp_degc
            tof_sensor_id = self.tof_sensor_id
            tof_distance_array_meters = self.tof_distance_array_meters.copy()
            tof_status_array = self.tof_status_array.copy()
            tof_xyz_array_meters_in_tof_frame = self.tof_xyz_array_meters_in_tof_frame.copy()
            tof_xyz_array_meters_in_car_frame = self.tof_xyz_array_meters_in_car_frame.copy()

            # Publish the heading angle
            heading_msg = Float32()
            heading_msg.data = heading_angle_in_radians * 180.0/np.pi
            self.imu_heading_angle_publisher_.publish(heading_msg)
            
            # =======================================
            # START OF: INSERT POLICY CODE BELOW HERE
            # =======================================

            # OBSERVATIONS:
            # > The "x_coords", "y_coords" and "cone_colour" variable are:
            #   - Lists with length equal to num_cones (the number of cones detected).
            #   - "x_coords" and "y_coords" give the x and y world coordinates of the cones respectively
            #   - "cone_colour" gives the colour of the cones (0 for yellow, 1 for blue)
            #   - A cone represents a single index of all three lists e.g. x_coords[i], y_coords[i], cone_colour[i] represent the ith cone

            # > The "heading_angle_in_radians" is:
            #   - The current heading angle of the car as measured by the IMU.
            #   - This value is in units of radians.
            #   - "Zero" heading angle will correspond to direction the car it facing when the IMU first boots up
            #   - The "re-zero" the IMU, there are class variables that set the current value as zero everytime
            #     that the "FSM_STATE_PUBLISHING_POLICY_ACTION" request is received.
            #   - A sensor fusion algorithms is running the IMU chip itself to estimate this heading angle
            #     based on observations of acceleration, angluar velocity, and magnetometer.
            #   - Hence, the heading angle can slowly drift with time.
            #   - The IMU chip itself "calibrates" for drift any time that the IMU is stationary.

            # > The "roll_angle_in_radians" and "pitch_angle_in_radians" is:
            #   - The current angles of the car as measured by the IMU.
            #   - These values are in units of radians.
            #   - "Zero" angle corresponds to when the car body is flat relative to gravity.
            #   - Roll  is the angle about the positive x-axis of the car, which points forwards.
            #   - Pitch is the angle about the positive y-axis of the car, which points left.

            # > The collection of Time-of-Flight (ToF) measurements
            #   > "tof_usec_since_last_msg"
            #     - The time between when the current measurement was taken and the once before it
            #     - This doesn't account for any extra time it took the measurement to get to this point in the code
            #   > "tof_silicon_temp_degc"
            #     - The temperature of the ToF sensor's chip
            #   > "tof_distance_array_meters"
            #     - A 1D numpy array of length 16 with the distance measurements in each zone
            #   > "tof_xyz_array_meters_in_tof_frame"
            #     - A 2D numpy array of size 3-by-16 with the (x,y,z) coordinates for each distance measurement
            #       in the frame of the tof
            #   > "tof_xyz_array_meters_in_car_frame"
            #     - A 2D numpy array of size 3-by-16 with the (x,y,z) coordinates for each distance measurement
            #       in the frame of the car.
            #     - This is the "tof_xyz_array_meters_in_tof_frame" measurements translated up by the hieigh of
            #       the ToF sensor above the ground, and rotate by the roll and pitch measurements from the IMU.
            #   > "tof_status_array"
            #     - A 1D numpy array of length 16 with the status of the measurement for each zone.
            #     - A status value of 5 or 9 is consider a good measurement, everything else is questionable.
            #     - The possible status values are:
            #       | Code | Description
            #       |   0  | Ranging data are not updated
            #       |   1  | Signal rate too low on SPAD array
            #       |   2  | Target phase
            #       |   3  | Sigma estimator too high
            #       |   4  | Target consistency failed
            #       |   5  | Range valid
            #       |   6  | Wrap around not performed (Typically the first range)
            #       |   7  | Rate consistency failed
            #       |   8  | Signal rate too low for the current target
            #       |   9  | Range valid with large pulse (may be due to a merged target)
            #       |  10  | Range valid, but no target detected at previous range
            #       |  11  | Measurement consistency failed
            #       |  12  | Target blurred by another one, due to sharpener
            #       |  13  | Target detected but inconsistent data. Frequently happens for secondary targets.)
            #       | 255  | No target detected (only if number of target detected is enabled)

            # ACTIONS:
            # > The "esc_action" is:
            #   - The action for driving the main motor (esc := electronic speed contrller).
            #   - In units of signed percent, with valid range [-100,100]
            # > The "steering_action" is:
            #   - The action for changing the position of the steering servo.
            #   - In units of signed percent, with valid range [-100,100]

            # ACRONYMS:
            # "esc" := Electronic Speed Controller
            #   - This device on the Traxxas car does NOT control the speed.
            #   - The "esc" device set the voltage to the motor within the
            #     range [-(max voltage),+(max voltage)]
    
            # ===============================================================================================================
            #                                               AMAZING POLICY
            # ===============================================================================================================

            # ---------------------------------------------------------------------------------------------------------------
            #                                           VARIABLES AND CONSTANTS
            # ---------------------------------------------------------------------------------------------------------------

            # + PID and Gain values
            P_STEERING = 0.2  # P value for Steering Action; SIMULATION=0.0005; DEFAULT=0.25
            I_STEERING = 0.0025  # I value for Steering Action; SIMULATION=0.000002; DEFAULT=0.001
            D_STEERING = 0.025  # D value for Steering Action; SIMULATION=0.000025; DEFAULT=0.0125
            HEADING_GAIN = 0.2  # + Gain for heading error correction; SIMULATION=0.0001; DEFAULT=0.05

            INTEGRAL_ADAPTION = 0.9

            # + Steering Adjustment Values
            STEER_OFFSET = 0  # Offset to adjust for biased steering; DETAULT=0
            FIXED_ESC_VALUE = 15.0  #15.0  # Fixed ESC value for steering tests; DEFAULT=33.0

            # + Lookahead Window Definition(in mm)
            X_MIN = 200.0  # Start of lookahead window; SIMULATION=500.0; DETAULT=50.0
            X_MAX = 2_000.0  # End of lookahead window; SIMULATION=8_000.0; DEFAULT=2_000.0
            X_LOOKAHEAD = 800.0  # Lookahead point (where we aim to be in the centre of the road); SIMULATION=1_000.0; DEFAULT=800.0
            ANGLE_THRESHOLD = 0.0  # Max heading angle (in degrees) to consider INVALID cone detections [-10°, +10°]; DETAULT=10.0

            # ---------------------------------------------------------------------------------------------------------------
            #                                               HELPER FUNCTIONS
            # ---------------------------------------------------------------------------------------------------------------

            # + Filter cones based on Lookahead Window and Front View Restrictions
            def filter_forward_cones(x_coords, y_coords, cone_colour):
                # Convert to arrays for convenience
                x_array = np.array(x_coords, dtype=float)  # Forward (in mm)
                y_array = np.array(y_coords, dtype=float)  # Lateral (in mm)
                c_array = np.array(cone_colour, dtype=int)  # Colour (0=yellow, 1=blue)

                # + Apply forward window (we only get the cones values within range)
                window_mask = (x_array >= X_MIN) & (x_array <= X_MAX)
                x_window, y_window, c_window = x_array[window_mask], y_array[window_mask], c_array[window_mask]

                # + Compute angle θ = atan2(y, x) for each cone within the window
                theta = np.arctan2(y_window, x_window)  # In radians

                # + Apply front view restriction (we only keep cones within the angle threshold)
                angle_mask = np.abs(np.degrees(theta)) > ANGLE_THRESHOLD

                return x_window[angle_mask], y_window[angle_mask], c_window[angle_mask]
            

            # + Compute dominant road direction using PCA
            def compute_PCA_road_direction(x, y):
                # + If enough info, we assume forwards direction
                if len(x) < 2:
                    return np.array([1.0, 0.0])  

                # + Prepare points for PCA
                pts = np.column_stack((x, y)).astype(float)
                mean = np.mean(pts, axis=0)
                centered = pts - mean
                
                # + Compute covariance matrix and its eigenvalues/vectors
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eig(cov)
                principal_axis = eigvecs[:, np.argmax(eigvals)]  # Principal axis corresponds to largest eigenvalue

                # + Ensure principal axis points forwards (positive x direction)
                if principal_axis[0] < 0:
                    principal_axis = -principal_axis

                return principal_axis / np.linalg.norm(principal_axis)  # Normalize
            

            # + Compute lateral error to target line at lookahead point
            def compute_lateral_error(x_left, y_left, x_right, y_right, y_pca_target):
                # + Check if we have valid cones on both sides
                has_left_cones, has_right_cones = len(x_left) > 0, len(x_right) > 0

                # + Find the nearest LEFT and RIGHT CONES near the LOOKAHEAD 
                x_left_distances = np.abs(np.array(x_left) - X_LOOKAHEAD)
                x_right_distances = np.abs(np.array(x_right) - X_LOOKAHEAD)
                
                if has_left_cones and has_right_cones:
                    # + Pick closest in x-forward to lookahead
                    x_left_closest  = np.argmin(x_left_distances)
                    x_right_closest = np.argmin(x_right_distances)
                    # + Compute center between the two cones
                    return 0.5 * (y_left[x_left_closest] + y_right[x_right_closest])
                else:
                    return y_pca_target  # Default to target line if insufficient cones

            # ---------------------------------------------------------------------------------------------------------------
            #                                                 POLICY LOGIC
            # ---------------------------------------------------------------------------------------------------------------

            # + FILTER the cones to only keep those in FRONT of the car within the lookahead window
            x, y, c = filter_forward_cones(x_coords, y_coords, cone_colour)

            if(len(x) > 0 and len(y) > 0):
                # + COMPUTE DOMINANT ROAD DIRECTION using PCA
                road_direction = compute_PCA_road_direction(x, y)

                # + SEPARATE BLUE cones (left) and YELLOW cones (right)
                x_left, y_left, x_right, y_right = x[c == 1], y[c == 1], x[c == 0], y[c == 0]

                # + Use PCA road direction to choose LOOKAHEAD TARGET POINT
                # x_pca_target = X_LOOKAHEAD * road_direction[0] <- Not needed
                y_pca_target = X_LOOKAHEAD * road_direction[1]

                # + COMPUTE LATERAL ERROR to the target line at LOOKAHEAD TARGET POINT
                lateral_error = compute_lateral_error(x_left, y_left, x_right, y_right, y_pca_target)

                # + COMPUTE ERRORS for STEERING ACTION
                heading_error = np.arctan2(road_direction[1], road_direction[0])  # HEADING ERROR to the road direction (in radians)
                derivative_error = (lateral_error - self.prev_lateral_error) / self.timer_period
                #self.integral_lateral_error += lateral_error * self.timer_period
                self.integral_lateral_error = (1 - INTEGRAL_ADAPTION) * self.integral_lateral_error + INTEGRAL_ADAPTION * lateral_error * self.timer_period

                # + STEERING AND ESC ACTIONS
                steering_action = -(
                    P_STEERING * lateral_error  # P term for lateral error
                    + I_STEERING * self.integral_lateral_error  # I term for lateral error
                    + D_STEERING * derivative_error  # D term for lateral error
                    + HEADING_GAIN * np.degrees(heading_error) + STEER_OFFSET)
                
                observations = {
                    "cone_detections_num_cones": np.array([msg.n], dtype=int),
                    "cone_detections_x": np.array(x_coords, dtype=float),
                    "cone_detections_y": np.array(y_coords, dtype=float),
                    "cone_detections_side_of_road": np.array(cone_colour, dtype=float),
                    "distance_to_closest_point": lateral_error,
                    "heading_angle_relative_to_line": heading_error,
                }
                steering_action += self.rl_model.predict(observations)

                esc_action = FIXED_ESC_VALUE  # Keep forward motion

                # + UPDATE previous lateral error
                self.prev_lateral_error = lateral_error
            else:
                steering_action = 0.0
                esc_action = 0.0

            # ===============================================================================================================
            #                                               END OF THE POLICY
            # ===============================================================================================================

            # Publish debug messages to be viewed in Foxglove
            if distance_to_target_line > -9998:
                debug1_msg = Float32()
                debug1_msg.data = distance_to_target_line
                self.debug1_publisher_.publish(debug1_msg)

            debug2_msg = Float32()
            debug2_msg.data = steering_action
            self.debug2_publisher_.publish(debug2_msg)

        # Prepare the message to send
        msg = EscAndSteeringPercent()
        msg.esc_percent = esc_action
        msg.steering_percent = steering_action

        # Publish the message
        self.policy_action_publisher_.publish(msg)

        # Log the string published for debugging purposes:
        #self.get_logger().info("[POLICY NODE] Published esc action = " + str(esc_action) + ", steering action = " + str(steering_action))



    # CALLBACK FUNCTION: for the FSM state transition request
    def fsm_transition_request_callback(self, msg):
        # Extract the requests FSM state from the message
        requested_state = msg.data

        # Check that the requested state is valid
        if requested_state not in LIST_OF_FSM_STATES:
            # Log that this occurred
            self.get_logger().info("[POLICY NODE] Received request to transition to an invalid state, requested_state = " + str(requested_state))

        if requested_state == FSM_STATE_PUBLISHING_POLICY_ACTION:
            self.heading_angle_for_tare = self.heading_angle_from_imu

        # Transition the state
        self.fsm_state = requested_state

        # Log the data received for debugging purposes:
        self.get_logger().info("[POLICY NODE] Received request to transition to FSM state: " + str(msg.data))



    # CALLBACK FUNCTION: for the timer
    def timer_callback(self):

        # Convert the FSM state to a string
        state_as_string = "Unknown state"

        if (self.fsm_state == FSM_STATE_NOT_PUBLISHING_ACTIONS):
            state_as_string = "Not publishing any actions"
        elif (self.fsm_state == FSM_STATE_PUBLISHING_POLICY_ACTION):
            state_as_string = "Publishing policy actions"
        elif (self.fsm_state == FSM_STATE_PUBLISHING_ZERO_ACTIONS):
            state_as_string = "Publishing zero actions"

        # Prepare the message to send
        msg = String()
        msg.data = state_as_string

        # Publish the message
        self.fsm_state_publisher_.publish(msg)



    # CALLBACK FUNCTION: for receiving IMU data
    def imu_callback(self, msg):
        # Extract the heading angle into the class variable
        self.heading_angle_from_imu = msg.yaw
        self.pitch_angle_from_imu   = msg.pitch
        self.roll_angle_from_imu    = msg.roll
        # Print for debugging
        #roll_deg  = np.degrees(msg.roll)
        #pitch_deg = np.degrees(msg.pitch)
        #yaw_deg   = np.degrees(msg.yaw)
        #self.get_logger().info(f"[POLICY NODE] IMU Callback received (r,p,y) = ( {roll_deg:4.0f}, {pitch_deg:4.0f}, {yaw_deg:4.0f})")

    # CALLBACK FUNCTION: for receiving VL53L5CX ToF data
    def tof_callback(self, msg):
        # Extract the details and measurements
        self.tof_usec_since_last_msg = msg.usec_since_last_msg
        self.tof_sensor_id = msg.sensor_id
        self.tof_silicon_temp_degc = msg.silicon_temp_degc
        self.tof_distance_array_meters = np.asarray(msg.distance_mm, dtype=np.float32) * 1e-3
        self.tof_status_array = np.asarray(msg.target_status, dtype=np.uint8)
        # Convert to (x,y,z) coordinates
        self.tof_xyz_array_meters_in_tof_frame = self.distances_to_xyz_center_rays(distances = self.tof_distance_array_meters,Nx=4, Ny=4, Fx_deg=45.0, Fy_deg=45.0)
        # Transform to body frame coordinates
        if (self.tof_should_use_roll_and_pitch_for_rotation_into_car_frame):
            pitch_for_transform = self.pitch_angle_from_imu
            roll_for_transform  = self.roll_angle_from_imu
        else:
            pitch_for_transform = 0.0
            roll_for_transform  = 0.0
        self.tof_xyz_array_meters_in_car_frame = self.transform_tof_to_world(
            xyz_tof=self.tof_xyz_array_meters_in_tof_frame,
            t_world=self.tof_translation_vector_meters_in_car_frame,
            yaw=0.0,
            pitch=pitch_for_transform,
            roll=roll_for_transform,
        )
        # Call the function to publish as a point cloud
        self.publish_tof_pointcloud()



    def publish_tof_pointcloud(self):
        """Publish the latest ToF points as a PointCloud2 for visualization."""
        xyz = np.asarray(self.tof_xyz_array_meters_in_car_frame, dtype=np.float32)
        if xyz.size == 0:
            return
        if xyz.ndim != 2 or xyz.shape[0] != 3:
            self.get_logger().warning("[POLICY NODE] Unexpected ToF XYZ shape, skipping point cloud publish")
            return
        mask = np.isfinite(xyz).all(axis=0)
        if not np.any(mask):
            return
        points = xyz[:, mask].T
        points_list = points.tolist()
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.tof_pointcloud_frame_id
        cloud_msg = create_cloud_xyz32(header, points_list)
        self.tof_pointcloud_publisher_.publish(cloud_msg)



    def publish_transforms(self):
        """Broadcast dynamic transforms for the car and ToF frames."""
        now = self.get_clock().now().to_msg()

        # World -> car_frame using yaw heading
        car_tf = TransformStamped()
        car_tf.header.stamp = now
        car_tf.header.frame_id = self.world_frame_id
        car_tf.child_frame_id = self.car_frame_id
        car_tf.transform.translation.x = 0.0
        car_tf.transform.translation.y = 0.0
        car_tf.transform.translation.z = 0.0
        q_car = self.euler_to_quaternion(0.0, 0.0, self.heading_angle_from_imu)
        car_tf.transform.rotation.x = q_car[0]
        car_tf.transform.rotation.y = q_car[1]
        car_tf.transform.rotation.z = q_car[2]
        car_tf.transform.rotation.w = q_car[3]

        # car_frame -> tof_frame using IMU roll/pitch and known translation
        tof_tf = TransformStamped()
        tof_tf.header.stamp = now
        tof_tf.header.frame_id = self.car_frame_id
        tof_tf.child_frame_id = self.tof_pointcloud_frame_id
        translation = np.asarray(self.tof_translation_vector_meters_in_car_frame, dtype=float).reshape(3)
        tof_tf.transform.translation.x = float(translation[0])
        tof_tf.transform.translation.y = float(translation[1])
        tof_tf.transform.translation.z = float(translation[2])
        if self.tof_should_use_roll_and_pitch_for_rotation_into_car_frame:
            roll = float(self.roll_angle_from_imu)
            pitch = float(self.pitch_angle_from_imu)
        else:
            roll = 0.0
            pitch = 0.0
        q_tof = self.euler_to_quaternion(roll, pitch, 0.0)
        tof_tf.transform.rotation.x = q_tof[0]
        tof_tf.transform.rotation.y = q_tof[1]
        tof_tf.transform.rotation.z = q_tof[2]
        tof_tf.transform.rotation.w = q_tof[3]

        self.tf_broadcaster.sendTransform([car_tf, tof_tf])

    @staticmethod
    def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
        """Convert Euler angles (roll, pitch, yaw) to a quaternion."""
        half_roll = 0.5 * roll
        half_pitch = 0.5 * pitch
        half_yaw = 0.5 * yaw

        cr = np.cos(half_roll)
        sr = np.sin(half_roll)
        cp = np.cos(half_pitch)
        sp = np.sin(half_pitch)
        cy = np.cos(half_yaw)
        sy = np.sin(half_yaw)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return (qx, qy, qz, qw)


    # =============================================================================
    # > Description of the VL53L5CX ToF sensor array measurement
    #   - The documentation can be found here: https://cdn.sparkfun.com/assets/6/e/3/0/6/vl53l5cx-datasheet.pdf
    #   - The key description of the array is on Page 19, Section 5.1.3 "Effective zone orientation", Figure 19.
    #   - Consider the following "ASCII art" to explain:
    #
    #     +-----+-----+-----+-----+ ←--- 22.500 deg (vertical half-FoV edge)
    #     |  0  |  1  |  2  |  3  | ←--- 16.875 deg (row center elevation)
    #     +-----+-----+-----+-----+ ←--- 11.250 deg
    #     |  4  |  5  |  6  |  7  | ←---  5.625 deg
    #     +-----+-----+-----+-----+ ←---  0.000 deg (optical axis)
    #     |  8  |  9  | 10  | 11  |
    #     +-----+-----+-----+-----+
    #     | 12  | 13  | 14  | 15  |
    #     +-----+-----+-----+-----+         ↑
    #                 ↑     ↑     ↑          --- vertical (elevation) angles φ
    #                 |     |     |
    #                 0   11.25 22.50 deg   ←--- horizontal (azimuth) angles θ (half-FoV tick marks from center)
    #
    #     FIGURE: "ASCII art" of the zone indexing as per the measurement array,
    #             and the angle (relative to the sensor) at the center or boundary
    #             of each zone.
    #
    #   - Consider that as you look at this "ASCII art" you are positioned
    #     on the car at the ToF sensor and looking outwards from the sensor.
    #   - The numbers in the grid are the indices in the 1D measurement arrays.
    #   - Hence index 0 is the **top-left** measurement as seen from the sensor.
    #   - The angular span of the field-of-view (FoV) is described on Page 3, Section 1.2 "Field of view".
    #   - The FoV is approximately 45 degrees for both horizontal and vertical.
    #     Consequently, each cell spans ~11.25° in both azimuth and elevation for a 4×4 grid.
    #
    # -----------------------------------------------------------------------------
    # > Axes and projection model
    #   - ToF/sensor frame axes: x forward (out of sensor), y left, z up.
    #   - Center-ray pinhole model per cell:
    #       (a) Cell center at azimuth θ (left positive) and elevation φ (up positive)
    #       (b) Define an unnormalized ray d = [1, tan(θ), tan(φ)]^T
    #       (c) Normalize to û = d/||d||
    #       (d) Scale by the measured range R: p = R * û.
    #
    # -----------------------------------------------------------------------------
    # > IMU orientation conventions & how we consume them
    #   - Your FSM300 reports yaw, pitch, roll such that:
    #       (y) Positive yaw:   rotate +about WORLD z (up)
    #       (p) Positive pitch: rotate +about subsequent y
    #       (r) Positive roll:  rotate +about subsequent x
    #     This corresponds to an **intrinsic** yaw–pitch–roll composition that maps
    #     world vectors into the ToF frame:
    #         R_from_world_to_tof = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    #
    #   - In many robotics codebases, it’s convenient to specify orientation as
    #     **intrinsic yaw–pitch–roll** (i.e., about the body / ToF axes, applied in the
    #     order yaw→pitch→roll) for the rotation **from world to ToF**:
    #         R_world_to_tof (intrinsic ZYX) = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    #
    #   - This rotation matrix would be used as:
    #         vector_in_tof_frame = R_world_to_tof @ vector_in_world_frame
    #
    #   - As the ToF distances give us vectors in the ToF frame, we need the "reverse"
    #     rotation to tranform to vectors into the world frame.
    #   - The "reverse" of a rotation matrix is just the tanspose, and for the construction
    #     we have, this becomes:
    #         R_tof_to_world = Rz(-yaw) @ Ry(-pitch) @ Rx(-roll)
    #   - Hence, this rotation matrix would be used as:
    #         vector_in_world_frame = R_tof_to_world @ vector_in_tof_frame
    #
    # -----------------------------------------------------------------------------
    # > What the functions below do
    #
    # 1) distances_to_xyz_center_rays(distances, Nx=4, Ny=4, Fx_deg=45, Fy_deg=45)
    #    - INPUT:
    #        distances: 1D numpy array of length Ny*Nx with range measurements [meters].
    #                   distances[i] is the measured range for zone i (0..Ny*Nx-1).
    #        Nx, Ny   : number of columns and rows (defaults to 4×4).
    #        Fx_deg   : horizontal FoV in degrees (across columns).
    #        Fy_deg   : vertical FoV in degrees (across rows).
    #
    #    - AXIS CONVENTION (sensor/ToF frame):
    #        x : straight out of the sensor (forward)
    #        y : to the LEFT of the sensor
    #        z : UP from the sensor
    #      With this convention, zone index 0 (top-left) corresponds to positive (x, y, z).
    #
    #    - METHOD (center-ray “pinhole” model):
    #      For each cell center we assign azimuth θ (left is +) and elevation φ (up is +):
    #          θ(c) = (0.5 - (c + 0.5)/Nx) * Fx
    #          φ(r) = (0.5 - (r + 0.5)/Ny) * Fy
    #      measured in radians.
    #      The (unnormalized) ray direction in the ToF frame is:
    #          d = [ 1,  tan(θ),  tan(φ) ]
    #      Normalize to a unit vector û = d / ||d|| and scale by the measured range R:
    #          p = R * û
    #      This produces symmetric behavior in y and z and correctly interprets R as
    #      “range along the ray”.
    #
    #    - OUTPUT:
    #        A 2D numpy array of shape (3, Ny*Nx) whose columns are [x; y; z] for each zone.
    #        If a distance is NaN or <= 0, the corresponding column is set to NaNs.
    #
    # 2) transform_tof_to_world(xyz_tof, t_world, yaw, pitch, roll)
    #    - PURPOSE:
    #        Transform the points from the ToF frame into a world frame using a given
    #        translation (expressed in the world frame) and yaw/pitch/roll intrinsic Euler angles
    #        that rotate the world frame into the ToF frame.
    #
    #    - CONVENTIONS:
    #        * Euler angles (yaw, pitch, roll) are in radians.
    #        * Axes for both ToF and world share the same handedness and axis naming:
    #              x forward, y left, z up
    #        * Rotation order:        R_world_to_tof = Rz(roll) @ Ry( pitch) @ Rx( yaw)
    #        * Hence, rotation order: R_tof_to_world = Rz(-yaw) @ Ry(-pitch) @ Rx(-roll)
    #          i.e., apply negative roll about x, then negative pitch about y, then negative
    #          yaw about z to rotate a vector from ToF frame into the world frame.
    #        * Translation t_world = [tx, ty, tz] is expressed in the world frame and is added after rotation:
    #              p_world = R * p_tof + t_world
    #
    #    - INPUT:
    #        xyz_tof : array (3, N), columns are [x; y; z] in ToF frame
    #        t_world : array-like (3,), translation in world frame
    #        yaw     : convention as described above, radians
    #        pitch   : convention as described above, radians
    #        roll    : convention as described above, radians
    #
    #    - OUTPUT:
    #        xyz_world: array (3, N)
    #
    # -----------------------------------------------------------------------------
    # > Notes
    #   - This center-ray approach is the common and robust choice for projecting
    #     coarse angular bins to 3D points from range. If you require a point that
    #     represents the *area* of the bin more precisely, you could average the four
    #     corner rays (patch centroid). For a 45° FoV split into 4×4, the difference
    #     from the center-ray point is typically small (millimeters per meter of range).
    # =============================================================================


    @staticmethod
    def distances_to_xyz_center_rays(distances: np.ndarray,
                                    Nx: int = 4,
                                    Ny: int = 4,
                                    Fx_deg: float = 45.0,
                                    Fy_deg: float = 45.0) -> np.ndarray:
        """
        Convert a 1D array of ToF ranges (VL53L5CX-style 4x4 grid) into 3×N XYZ points
        in the ToF frame using center-ray projection, with axes:
            x forward (out of sensor), y left, z up.

        Parameters
        ----------
        distances : np.ndarray
            1D array of length Ny*Nx with range values in meters.
            Indexing is row-major with index 0 at the top-left cell.
        Nx, Ny : int
            Number of columns and rows. Defaults 4×4.
        Fx_deg, Fy_deg : float
            Horizontal and vertical field of view in degrees.

        Returns
        -------
        xyz : np.ndarray
            Array of shape (3, Ny*Nx). Column i is the [x; y; z] point for distances[i].
            NaNs are returned for invalid/non-positive ranges.
        """
        distances = np.asarray(distances).reshape(-1)
        assert distances.size == Nx * Ny, f"Expected {Nx*Ny} distances, got {distances.size}"

        # Build row/col index arrays for each linear index (row-major: 0..Ny*Nx-1)
        idx = np.arange(Nx * Ny)
        rows = idx // Nx  # 0..Ny-1 (top to bottom)
        cols = idx % Nx   # 0..Nx-1 (left to right)

        # Angles at cell centers: left/up are positive by construction
        Fx = np.deg2rad(Fx_deg)
        Fy = np.deg2rad(Fy_deg)
        # θ(c): azimuth (around +z), positive to the LEFT
        theta = (0.5 - (cols + 0.5) / Nx) * Fx
        # φ(r): elevation (around -y), positive UP
        phi   = (0.5 - (rows + 0.5) / Ny) * Fy

        # Pinhole-style direction before normalization: d = [1, tanθ, tanφ]
        tan_theta = np.tan(theta)
        tan_phi   = np.tan(phi)
        dx = np.ones_like(tan_theta)
        dy = tan_theta
        dz = tan_phi

        # Normalize to unit rays
        norms = np.sqrt(dx*dx + dy*dy + dz*dz)
        ux = dx / norms
        uy = dy / norms
        uz = dz / norms

        # Scale by ranges; invalid ranges -> NaNs
        R = distances.astype(float)
        mask_valid = np.isfinite(R) & (R > 0.0)
        xyz = np.vstack((ux * R, uy * R, uz * R))
        xyz[:, ~mask_valid] = np.nan
        return xyz


    @staticmethod
    def transform_tof_to_world(xyz_tof: np.ndarray,
                            t_world: np.ndarray,
                            yaw: float,
                            pitch: float,
                            roll: float) -> np.ndarray:
        """
        Apply a rigid transform to points from the ToF frame into the world frame.

        Conventions:
        - ToF/world axes: x forward, y left, z up.
        - Euler angles rotate  World -> ToF with order: Rz(roll) @ Ry( pitch) @ Rx( yaw).
        - Hence, angles rotate ToF -> World with order: Rz(-yaw) @ Ry(-pitch) @ Rx(-roll).
        - Translation is expressed in the world frame and added after rotation.

        Parameters
        ----------
        xyz_tof : np.ndarray
            (3, N) array of points in the ToF frame.
        t_world : array-like of length 3
            Translation vector [tx, ty, tz] in the world frame.
        yaw : float
            Convention as decribed above, in radians.
        pitch : float
            Convention as decribed above, in radians.
        roll : float
            Convention as decribed above, in radians.

        Returns
        -------
        xyz_world : np.ndarray
            (3, N) array of transformed points in the world frame.
        """
        xyz_tof = np.asarray(xyz_tof, dtype=float)
        assert xyz_tof.ndim == 2 and xyz_tof.shape[0] == 3, "xyz_tof must be shape (3, N)"

        # Rotation matrices
        cy, sy = np.cos(-yaw),   np.sin(-yaw)
        cp, sp = np.cos(-pitch), np.sin(-pitch)
        cr, sr = np.cos(-roll),  np.sin(-roll)

        # Rz(yaw)
        Rz = np.array([[ cy, -sy,  0.0],
                    [ sy,  cy,  0.0],
                    [0.0, 0.0,  1.0]])

        # Ry(pitch)
        Ry = np.array([[  cp, 0.0,  sp],
                    [ 0.0, 1.0, 0.0],
                    [ -sp, 0.0,  cp]])

        # Rx(roll)
        Rx = np.array([[1.0, 0.0, 0.0],
                    [0.0,  cr, -sr],
                    [0.0,  sr,  cr]])

        # Combined rotation: ToF -> World
        R = Rz @ Ry @ Rx

        # Apply transform
        t = np.asarray(t_world, dtype=float).reshape(3, 1)
        xyz_world = (R @ xyz_tof) + t
        return xyz_world



def main(args=None):
    # Initialise ROS2 for this script
    rclpy.init(args=args)
    # Start as instance of the PolicyNode class
    node = PolicyNode()
    # Enter a ROS2 spin
    rclpy.spin(node)
    # Shutdown the nodes in this script
    rclpy.shutdown()

if __name__ == '__main__':
    main()
