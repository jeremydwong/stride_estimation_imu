import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, Dict, Any

# Note: For better peak detection, you can uncomment the following line and use scipy's find_peaks:
# from scipy.signal import find_peaks

class ComputePos:
    """IMU position computation using zero velocity updates"""
    
    def __init__(self):
        # Persistent variables for kf_tilt
        self.kf_P = None
        self.kf_Q = None
        self.kf_R = None
    
    def compute_pos(self, W: np.ndarray, A: np.ndarray, PERIOD: float, 
                   USE_KF: bool = True, W_FF: Optional[float] = None, 
                   A_FF: Optional[float] = None, T_FF: Optional[float] = None,
                   MAX_T_FF: Optional[float] = None, FF: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes IMU positions assuming zero velocity updates
        
        Args:
            W: Angular Velocity, finite difference form (rad/sec * PERIOD)
            A: Acceleration (m/s/s)
            PERIOD: Sampling Period (seconds)
            USE_KF: Use Tilt Kalman Filter (default True)
            W_FF: Upper Threshold for Angular Velocity (deg/sec) for footfalls
            A_FF: Upper Threshold for (Acceleration minus Gravity) (m/s/s) for footfalls
            T_FF: Minimum Time between footfalls (default 0.4 sec)
            MAX_T_FF: Maximum Time during rest periods (default 3*T_FF)
            FF: Logical array indicating footfalls (to override internal detection)
        
        Returns:
            Dictionary containing results
        """
        # Inertial navigation mechanization
        N = W.shape[0]
        t = np.arange(1, N+1) * PERIOD
        
        # Compute tilt based on accelerometer readings
        accel_phi, accel_theta = self.acc_tilt(A)
        
        # Determine footfalls
        if FF is None:
            FF, stationary_periods = self.foot_fall(W, A, PERIOD, W_FF, A_FF, T_FF, MAX_T_FF)
        else:
            # If FF is provided, create stationary_periods from it
            stationary_periods = np.zeros(N, dtype=bool)
            # This is a simplified version - you might need to adjust based on your needs
            stationary_periods[FF] = True
        
        result = {'FF': FF}
        
        # Initialize variables
        quaternion = np.zeros((N, 4))
        An = np.zeros((N, 3))
        Anz = np.zeros((N, 3))
        
        # Initialize quaternions
        quaternion[0, :] = self.kf_tilt(PERIOD)
        
        # Initialize for ZUPT
        last_footfall = 0
        
        for i in range(1, N):
            # Compute attitude using quaternion representation
            quaternion[i, :] = self.qua_est(W[i, :], quaternion[i-1, :])
            
            # Transform accelerations from body to navigation frame
            rotation_matrix = self.qua2rot(quaternion[i, :])
            An[i, :] = rotation_matrix @ A[i, :]
            
            if USE_KF:
                # Apply KF compensation on tilt
                quaternion[i, :] = self.kf_tilt(PERIOD, quaternion[i, :], 
                                               accel_theta[i], accel_phi[i], 
                                               stationary_periods[i])
            
    
            # Apply Zero Velocity Updates (ZUPT)
            if i == 2:
                last_footfall = 0

            if FF[i]:
                step_range = np.arange(last_footfall + 1, i + 1)
                # Skip short duration FF
                if len(step_range) >= 2:
                    step_samples = len(step_range)
                    # Compute final error
                    velocity_error = np.sum(An[step_range, :], axis=0)
                    # Compute the accelerometer error assuming it is linear
                    acceleration_error = velocity_error / step_samples
                    # Apply error corrections in accelerations
                    Anz[step_range, :] = An[step_range, :] - acceleration_error
                    last_footfall = i
        
        result['Anz'] = Anz
        result['An'] = An
        result['A'] = A
        result['W'] = W
        result['quaternion'] = quaternion
        
        # Convert to Euler angles
        euler = self.qua2eul(quaternion)
        result['euler'] = euler
        
        # Plot attitude
        plt.figure()
        plt.plot(t, np.degrees(euler))
        plt.hold = True
        plt.grid(True)
        plt.ylabel('Euler angles [deg]')
        plt.xlabel('time [s]')
        plt.legend(['Roll', 'Pitch', 'Heading'])
        plt.plot(t[FF], np.degrees(euler[FF, :]), '*')
        
        # Compute velocity and position
        V = np.cumsum(Anz, axis=0) * PERIOD
        Vm = np.sqrt(np.sum(V[:, 0:2]**2, axis=1))
        result['V'] = V
        result['Vm'] = Vm
        
        # Use speed information to determine the walking section
        try:
            result['FF_walking'], result['FF_max_speed'] = self.detect_walking_section(result)
        except:
            result['FF_walking'] = result['FF']
            result['FF_max_speed'] = np.zeros_like(result['FF'])
        
        # Plot velocity
        plt.figure()
        plt.plot(t, V)
        plt.plot(t, Vm, 'k')
        plt.hold = True
        plt.grid(True)
        plt.ylabel('V [m/s]')
        plt.xlabel('time [s]')
        plt.plot(t[FF], Vm[FF], '*g')
        plt.plot(t[result['FF_walking']], Vm[result['FF_walking']], '.k')
        plt.legend(['Vx', 'Vy', 'Vz', '|V|', 'Footfall', 'Footfall during walk'])
        
        # Compute and plot positions
        P = np.cumsum(V, axis=0) * PERIOD
        result['P'] = P
        
        # 3D position plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(P[:, 0], -P[:, 1], -P[:, 2])
        ax.plot(P[FF, 0], -P[FF, 1], -P[FF, 2], '.k')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.grid(True)
        ax.axis('equal')
        
        plt.show()
        
        return result
    
    def foot_fall(self, W: np.ndarray, A: np.ndarray, PERIOD: float,
                  W_FF: Optional[float] = None, A_FF: Optional[float] = None,
                  T_FF: Optional[float] = None, MAX_T_FF: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Detect foot falls based on angular velocity and acceleration thresholds"""
        
        PLOT_DETAILS = True
        GRAVITY = 9.80297286843
        
        # FF determination settings
        if W_FF is None:
            W_FF = 30  # Threshold rate (deg/sec)
        if A_FF is None:
            A_FF = 1  # Threshold acceleration (m/s/s)
        if T_FF is None:
            T_FF = int(0.4 / PERIOD)  # Minimum time between FFs in samples
        else:
            T_FF = int(T_FF / PERIOD)  # Convert seconds to samples
        if MAX_T_FF is None:
            MAX_T_FF = T_FF * 3  # Maximum rest period
        else:
            MAX_T_FF = int(MAX_T_FF / PERIOD)  # Convert seconds to samples
        
        # Calculate magnitudes
        Wm = np.sqrt(np.sum(W**2, axis=1)) * 180 / np.pi / PERIOD
        Am = np.sqrt(np.sum(A**2, axis=1)) - GRAVITY
        
        # Find low dynamic conditions
        low_motion = np.where((Wm < W_FF) & (np.abs(Am) < A_FF))[0]
        
        # stationary_periods is used by the tilt compensation KF
        N = W.shape[0]
        stationary_periods = np.zeros(N, dtype=bool)
        stationary_periods[low_motion] = True
        
        # Detect low dynamic segments separated at least T_FF
        if len(low_motion) == 0:
            FF = np.zeros(N, dtype=bool)
            FF[-1] = True
            return FF, stationary_periods
        
        diff_low_motion = np.diff(low_motion)
        valid_FF = np.where(diff_low_motion > T_FF)[0]
        
        if len(valid_FF) > 0:
            start_FF = low_motion[np.concatenate(([0], valid_FF + 1))]
            end_FF = low_motion[np.concatenate((valid_FF, [len(low_motion) - 1]))]
        else:
            start_FF = np.array([low_motion[0]])
            end_FF = np.array([low_motion[-1]])
        
        # Find best FF point
        FF_index = []
        for i in range(len(start_FF)):
            Wm_cut = Wm[start_FF[i]:end_FF[i]+1]
            N_cut = len(Wm_cut)
            
            if N_cut == 0:
                continue
                
            R_seg = N_cut // MAX_T_FF
            
            if R_seg > 0:
                # Segment long low dynamic intervals
                Wm_seg = Wm_cut[:R_seg*MAX_T_FF].reshape(MAX_T_FF, R_seg)
                min_idx = np.argmin(Wm_seg, axis=0)
                FF_cut = min_idx + np.arange(R_seg) * MAX_T_FF + start_FF[i]
                FF_index.extend(FF_cut)
            
            # Handle remainder
            if N_cut > R_seg * MAX_T_FF:
                min_idx2 = np.argmin(Wm_cut[MAX_T_FF*R_seg:]) + MAX_T_FF*R_seg
                FF_index.append(min_idx2 + start_FF[i])
        
        # Force the last sample to be a FF
        if len(FF_index) > 0:
            FF_index = np.array(FF_index) - 1  # Adjust for 0-based indexing
        FF_index = np.append(FF_index, N-1)
        
        # Create FF boolean vector
        FF = np.zeros(N, dtype=bool)
        FF[FF_index.astype(int)] = True
        
        # Make plots
        t = np.arange(1, N+1) * PERIOD
        
        plt.figure()
        
        if PLOT_DETAILS:
            plt.subplot(2, 1, 1)
            plt.plot(t[low_motion], Wm[low_motion], '.y')
            plt.plot(t[start_FF], Wm[start_FF], 'og')
            plt.plot(t[end_FF], Wm[end_FF], 'or')
            
            plt.subplot(2, 1, 2)
            plt.plot(t[low_motion], Am[low_motion], '.y')
            plt.plot(t[start_FF], Am[start_FF], 'og')
            plt.plot(t[end_FF], Am[end_FF], 'or')
        
        plt.subplot(2, 1, 1)
        plt.plot(t, Wm)
        plt.grid(True)
        plt.ylabel('W [deg/sec]')
        plt.title('Foot fall detection')
        plt.plot(t[FF], Wm[FF], '*k')
        plt.legend(['Signal', 'Foot-fall'])
        
        plt.subplot(2, 1, 2)
        plt.plot(t, Am)
        plt.grid(True)
        plt.ylabel('A [m/sec^2]')
        plt.plot(t[FF], Am[FF], '*k')
        plt.xlabel('t [sec]')
        
        return FF, stationary_periods
    
    def kf_tilt(self, PERIOD: float, quaternion: Optional[np.ndarray] = None, 
                accel_theta: Optional[float] = None, accel_phi: Optional[float] = None,
                is_stationary: bool = False) -> np.ndarray:
        """Tilt error compensation using accelerometer-based tilt"""
        
        if quaternion is None:
            # Initialize KF
            self.kf_P = 0  # Initial covariance error
            self.kf_Q = 1.5e-5 * PERIOD  # Process (gyros) covariance error
            self.kf_R = 1.5e-1 * PERIOD  # Measurement (accelerometer) covariance error
            return np.array([1, 0, 0, 0])
        
        # Propagate covariance error
        self.kf_P = self.kf_P + self.kf_Q
        
        if is_stationary:
            # Compute Kalman gain
            K = self.kf_P / (self.kf_P + self.kf_R)
            euler = self.qua2eul(quaternion.reshape(1, -1))[0]
            
            # Apply corrections to the state
            corrected_euler = [
                euler[0] - (euler[0] - accel_phi) * K,
                euler[1] - (euler[1] - accel_theta) * K,
                euler[2]
            ]
            quaternion = self.eul2qua(corrected_euler)
            
            # Update covariance error
            self.kf_P = (1 - K) * self.kf_P * (1 - K) + K * self.kf_R * K
        
        return quaternion
    
    def qua_est(self, W: np.ndarray, quaternion_prev: np.ndarray) -> np.ndarray:
        """Estimate quaternion from angular velocity"""
        mag = np.sqrt(np.sum(W**2))
        
        if mag != 0:
            sin_mag = np.sin(mag/2.0) / mag
        else:
            sin_mag = 0.5
        
        rotation = np.array([np.cos(mag/2.0), sin_mag*W[0], sin_mag*W[1], sin_mag*W[2]])
        
        a, b, c, d = quaternion_prev
        quaternion_sqw = np.array([
            [a, -b, -c, -d],
            [b,  a, -d,  c],
            [c,  d,  a, -b],
            [d, -c,  b,  a]
        ])
        
        quaternion = quaternion_sqw @ rotation
        return quaternion
    
    def qua2eul(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles"""
        if quaternion.ndim == 1:
            quaternion = quaternion.reshape(1, -1)
        
        a = quaternion[:, 0]
        b = quaternion[:, 1]
        c = quaternion[:, 2]
        d = quaternion[:, 3]
        
        phi = np.arctan2(2*(a*b + c*d), a**2 - b**2 - c**2 + d**2)
        the = np.arcsin(2*(a*c - d*b))
        psi = np.arctan2(2*(a*d + b*c), a**2 + b**2 - c**2 - d**2)
        
        euler = np.column_stack([phi, the, psi])
        return euler
    
    def qua2rot(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        a, b, c, d = quaternion
        
        rotation_matrix = np.array([
            [a**2 + b**2 - c**2 - d**2, 2*(b*c - a*d), 2*(b*d + a*c)],
            [2*(b*c + a*d), a**2 - b**2 + c**2 - d**2, 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), a**2 - b**2 - c**2 + d**2]
        ])
        
        return rotation_matrix
    
    # Missing functions that need to be implemented:
    
    def acc_tilt(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute tilt angles from accelerometer data.
        
        Returns:
            accel_phi: Roll angle from accelerometer
            accel_theta: Pitch angle from accelerometer
        """
        GRAVITY = 9.80297286843
        
        # Compute tilt from acceleration
        AX = A[:, 0]
        AY = A[:, 1]
        
        # Compute pitch
        accel_theta = AX / GRAVITY
        # Initialize with zeros
        accel_theta_result = np.zeros_like(accel_theta)
        valid_angles = np.abs(accel_theta) <= 1
        accel_theta_result[valid_angles] = np.arcsin(accel_theta[valid_angles])
        accel_theta = accel_theta_result
        
        # Compute roll angle
        accel_phi = AY / (np.cos(accel_theta) * GRAVITY)
        # Initialize with zeros
        accel_phi_result = np.zeros_like(accel_phi)
        valid_angles = np.abs(accel_phi) <= 1
        accel_phi_result[valid_angles] = -np.arcsin(accel_phi[valid_angles])
        accel_phi = accel_phi_result
        
        return accel_phi, accel_theta
    
    def eul2qua(self, euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to quaternion.
        
        Args:
            euler: [roll(phi), pitch(theta), yaw(psi)] in radians
        
        Returns:
            quaternion: [a, b, c, d] where a is the scalar part
        """
        if isinstance(euler, list):
            euler = np.array(euler)
        
        if euler.ndim == 1:
            phi = euler[0]
            the = euler[1]
            psi = euler[2]
        else:
            phi = euler[:, 0]
            the = euler[:, 1]
            psi = euler[:, 2]
        
        cos_phi = np.cos(phi/2.0)
        sin_phi = np.sin(phi/2.0)
        cos_the = np.cos(the/2.0)
        sin_the = np.sin(the/2.0)
        cos_psi = np.cos(psi/2.0)
        sin_psi = np.sin(psi/2.0)
        
        a = cos_phi * cos_the * cos_psi + sin_phi * sin_the * sin_psi
        b = sin_phi * cos_the * cos_psi - cos_phi * sin_the * sin_psi
        c = cos_phi * sin_the * cos_psi + sin_phi * cos_the * sin_psi
        d = cos_phi * cos_the * sin_psi - sin_phi * sin_the * cos_psi
        
        if euler.ndim == 1:
            return np.array([a, b, c, d])
        else:
            return np.column_stack([a, b, c, d])
    
    def detect_walking_section(self, walk_info: Dict[str, Any], MIN_WALK_SPEED: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect walking sections from velocity information.
        
        Args:
            walk_info: Dictionary containing FF and Vm arrays
            MIN_WALK_SPEED: Minimum foot velocity while walking (m/s)
        
        Returns:
            FF_walking: Boolean array of footfalls during walking
            FF_max_speed: Boolean array of velocity peaks
        """
        WALK_SPEED_PERCENTAGE = 0.90  # Determines the first and last foot fall
        
        FF_indices = np.where(walk_info['FF'])[0]
        Vm = walk_info['Vm']
        
        # Determine the most likely walking portion based on the MIN_WALK_SPEED value
        # Find peaks using simple method (scipy.signal.find_peaks equivalent)
        peaks_idx = self._find_peaks(Vm, MIN_WALK_SPEED)
        
        FF_max_speed = np.zeros(len(Vm), dtype=bool)
        if len(peaks_idx) > 0:
            FF_max_speed[peaks_idx] = True
        
        if len(peaks_idx) == 0:
            # No walking detected
            FF_walking = np.zeros(len(Vm), dtype=bool)
            return FF_walking, FF_max_speed
        
        median_vel = np.median(Vm[peaks_idx])
        likely_walk_sections = np.where(Vm > median_vel * WALK_SPEED_PERCENTAGE)[0]
        
        if len(likely_walk_sections) == 0:
            FF_walking = np.zeros(len(Vm), dtype=bool)
            return FF_walking, FF_max_speed
        
        # Find footfalls within the walking section
        footfall_mask = (FF_indices > likely_walk_sections[0]) & (FF_indices < likely_walk_sections[-1])
        footfall_in_walk = FF_indices[footfall_mask]
        
        FF_walking = np.zeros(len(Vm), dtype=bool)
        if len(footfall_in_walk) > 0:
            # Set all footfalls between first and last walking footfall
            start_idx = footfall_in_walk[0]
            end_idx = footfall_in_walk[-1]
            FF_walking[FF_indices[(FF_indices >= start_idx) & (FF_indices <= end_idx)]] = True
        
        # Determine if there are potential missdetected footfalls
        if np.sum(FF_walking) > 1:
            step_separation = np.diff(np.where(FF_walking)[0])
            if len(step_separation) > 0:
                median_step_separation = np.median(step_separation)
                missdetected_footfalls = np.abs(step_separation - median_step_separation) > median_step_separation/2
                if np.sum(missdetected_footfalls) > 0:
                    print(f'There are potential missdetected footfalls: {np.sum(missdetected_footfalls)}')
        
        return FF_walking, FF_max_speed
    
    def _find_peaks(self, data: np.ndarray, min_height: float) -> np.ndarray:
        """Simple peak finding algorithm"""
        peaks = []
        for i in range(1, len(data) - 1):
            if data[i] > min_height and data[i] > data[i-1] and data[i] > data[i+1]:
                peaks.append(i)
        return np.array(peaks)


# Example usage:
if __name__ == "__main__":
    # Create instance
    imu_processor = ComputePos()
    
    # Example data (you would load this from your HDF5 file)
    # W = ... # Angular velocity data
    # A = ... # Acceleration data  
    # PERIOD = 1/128.0  # Example: 128 Hz sampling rate
    
    # result = imu_processor.compute_pos(W, A, PERIOD)