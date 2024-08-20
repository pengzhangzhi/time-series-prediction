import numpy as np
from pykalman import KalmanFilter


class KalmanFilter:
    # Kalman filter implementation for data
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement_variance = estimated_measurement_variance

        #Initial
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0
    # Kalman filter update
    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        #Kalman gain
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        # Measurement update
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        # Error estimate update
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate
    # Kalman filter smooth
    def Kalman1D(observations,damping=1):
        # smoothed time series data
        observation_covariance = damping
        # initial value guess
        initial_value_guess = observations[0]
        transition_matrix = 1
        transition_covariance = 0.1
        initial_value_guess
        kf = KalmanFilter(
                initial_state_mean=initial_value_guess,
                initial_state_covariance=observation_covariance,
                observation_covariance=observation_covariance,
                transition_covariance=transition_covariance,
                transition_matrices=transition_matrix
            )
        # Kalman filter smooth
        pred_state, state_cov = kf.smooth(observations)
        return pred_state
    

    # #kalman input: y_pred-1,model_prediction       kalman output:label y_pred
    # kalman_outputs = []
    # for i in range(len(outputs_scaled_back)):
    #     kalman_out = kf.update(measurement=y_batch_scaled_back[i, -1].item(), prediction=outputs_scaled_back[i].item())
    #     mod_out = outputs[i].clone().detach() + 0 * outputs[i]
    #     mod_out[0] = kalman_out
    #     kalman_outputs.append(mod_out)
    # kalman_outputs = torch.cat(kalman_outputs).to("cuda")
