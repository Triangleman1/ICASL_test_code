import numpy as np               # Importing numpy for numerical operations, particularly with arrays
import pandas as pd              # Importing pandas as pd for data manipulation and analysis
class BatteryEKF:
    def __init__(self, df_charge, df_discharge, Capacity):
        # Extract LUTs from the dataframes
        self.soc_charge_lut = df_charge["SOC"].values
        self.r0_charge_lut = df_charge["R0"].values
        self.r1_charge_lut = df_charge["R1"].values
        self.c1_charge_lut = df_charge["C1"].values
        self.r2_charge_lut = df_charge["R2"].values
        self.c2_charge_lut = df_charge["C2"].values

        self.soc_discharge_lut = df_discharge["SOC"].values
        self.r0_discharge_lut = df_discharge["R0"].values
        self.r1_discharge_lut = df_discharge["R1"].values
        self.c1_discharge_lut = df_discharge["C1"].values
        self.r2_discharge_lut = df_discharge["R2"].values
        self.c2_discharge_lut = df_discharge["C2"].values

        self.Q_cap = Capacity * 3600 # Ah to Ampere-seconds
        self.p_charge = np.poly1d(np.polyfit(self.soc_charge_lut, df_charge["OCV"].values, 5))
        self.p_charge_derivative = np.polyder(self.p_charge)
        self.p_discharge = np.poly1d(np.polyfit(self.soc_discharge_lut, df_discharge["OCV"].values, 5))
        self.p_discharge_derivative = np.polyder(self.p_discharge)


        # State: [SOC, Vc1, Vc2]
        self.x = np.array([[1.0], [0.0], [0.0]]) 
        self.P = np.eye(3) * .001
        self.charge_state = 'discharging'
        
        # Process noise usually scales with time, but keeping it simple for now
        self.Q = np.diag([1e-3, 1e-7, 1e-7]) 
        self.R = 1e-5

        self.voltage_history = []
        self.eoc_window = 10        # samples to track
        self.eoc_threshold = -0.010 # 10mV drop from peak triggers EOC
        self.is_fully_charged = False

    def update_charge_state(self, current, threshold=0.05):
        # Only update state when current is significant
        # During rest (current ≈ 0), state is preserved from last active period
        if -current > threshold:
            self.charge_state = 'charging'
        elif -current < -threshold:
            self.charge_state = 'discharging'

    def initialize_soc(self, measured_voltage, soc_points, ocv_points):
        """
        Call this ONCE before starting the loop to find a starting SOC.
        """
        # Use linear interpolation to find SOC from Voltage
        # Note: interp expects the x-coordinates (ocv) to be increasing
        initial_soc = np.interp(measured_voltage, ocv_points[::1], soc_points[::1])
        
        # Update the state vector
        self.x[0, 0] = initial_soc
        return initial_soc
    def get_parameters(self):
        current_soc = np.clip(self.x[0, 0], 0, 1)
        
        # Interpolate each parameter based on current SOC
        if self.charge_state == 'charging':  # charging LUTs
            r0 = np.interp(current_soc, self.soc_charge_lut, self.r0_charge_lut)
            r1 = np.interp(current_soc, self.soc_charge_lut, self.r1_charge_lut)
            c1 = np.interp(current_soc, self.soc_charge_lut, self.c1_charge_lut)
            r2 = np.interp(current_soc, self.soc_charge_lut, self.r2_charge_lut)
            c2 = np.interp(current_soc, self.soc_charge_lut, self.c2_charge_lut)
        else:  # discharging LUTs
            r0 = np.interp(current_soc, self.soc_discharge_lut, self.r0_discharge_lut)
            r1 = np.interp(current_soc, self.soc_discharge_lut, self.r1_discharge_lut)
            c1 = np.interp(current_soc, self.soc_discharge_lut, self.c1_discharge_lut)
            r2 = np.interp(current_soc, self.soc_discharge_lut, self.r2_discharge_lut)
            c2 = np.interp(current_soc, self.soc_discharge_lut, self.c2_discharge_lut)
            
        return r0, r1, c1, r2, c2
    def get_OCV(self, soc):
        # Ensure SOC stays in bounds [0, 1]
        soc = np.clip(soc, 0, 1)
        if self.charge_state == 'charging':
            return self.p_charge(soc)
        else:  # discharging
            return self.p_discharge(soc)

    # Derivative for the EKF Jacobian (H matrix)
    def get_dOCV_dSOC(self, soc):
        #p_derivative = np.polyder(p)
        if self.charge_state == 'charging':
            return self.p_charge_derivative(soc)
        else:  # discharging
            return self.p_discharge_derivative(soc)


    def predict(self, current, dt):
        # Fetch dynamic parameters for the current SOC
        r0, r1, c1, r2, c2 = self.get_parameters()
        
        exp1 = np.exp(-dt / (r1 * c1))
        exp2 = np.exp(-dt / (r2 * c2))
        
        A = np.array([[1, 0, 0], [0, exp1, 0], [0, 0, exp2]])
        B = np.array([[-dt / self.Q_cap], [r1 * (1 - exp1)], [r2 * (1 - exp2)]])
        
        self.x = A @ self.x + B * current
        lambda_factor = 1.001  # small forgetting factor
        self.P = lambda_factor * (A @ self.P @ A.T) + (self.Q * dt)
        if self.charge_state == 'charging':  # charging
                self.x[0, 0] = np.clip(self.x[0, 0], 0, 1.05)  # allow slight overcharge
        else:
            self.x[0, 0] = np.clip(self.x[0, 0], 0, 1.0)

        # self.P = A @ self.P @ A.T + (self.Q * dt)

    def update(self, measured_voltage, current, rest_duration):
        # 2. Observation (h)
        # Vt = OCV(SOC) - I*R0 - Vc1 - Vc2
        self.update_charge_state(current)
        r0, r1, c1, r2, c2 = self.get_parameters()

        if current==0:
            # Fraction of relaxation complete (0=just stopped, 1=fully relaxed)
            relaxation = 1 - np.exp(-rest_duration / (r1 * c1))
            # Scale measurement trust by how relaxed it is
            self.R = 1e-3 / (relaxation + 1e-3)  # high R when unrelaxed, low when relaxed
            self.Q = np.diag([1e-7, 1e-6, 1e-6])
        else:
            self.R = 1e-2  # during discharge, trust voltage less
            self.Q = np.diag([1e-3, 1e-7, 1e-7])
        soc_val = self.x[0, 0]


        v_predicted = self.get_OCV(soc_val) - current * r0 - self.x[1, 0] - self.x[2, 0]

        r0, r1, c1, r2, c2 = self.get_parameters()

        # 3. Jacobian (H) - Derivative of h with respect to x
        H = np.array([[self.get_dOCV_dSOC(soc_val), -1, -1]])
        
        # 4. Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # 5. Update State and Covariance
        innovation = measured_voltage - v_predicted
        self.x = self.x + K * innovation
        self.P = (np.eye(3) - K @ H) @ self.P
        self.check_full_charge(measured_voltage, current)
        return self.x[0, 0] # Return estimated SOC

    def check_full_charge(self, voltage, current):
        # Only run during charging
        if -current <= 0:
            self.voltage_history = []  # reset when not charging
            return False
        
        self.voltage_history.append(voltage)
        
        # Need enough history to detect a peak
        if len(self.voltage_history) < self.eoc_window:
            return False
        
        # Keep window rolling
        if len(self.voltage_history) > self.eoc_window:
            self.voltage_history.pop(0)
        
        peak_voltage = max(self.voltage_history)
        dV = voltage - peak_voltage
        
        if dV < self.eoc_threshold:
            # Recalibrate SOC to 1.0 with high confidence
            self.x[0, 0] = 1.0
            self.P[0, 0] = 1e-6
            self.is_fully_charged = True
            return True
        
        return False