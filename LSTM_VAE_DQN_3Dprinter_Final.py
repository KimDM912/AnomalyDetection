# 라이브러리 import
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, 
    precision_score,
    recall_score,
    accuracy_score
)
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import matplotlib.pyplot as plt


# 데이터 경로 설정
base_path = r"C:\Users\User\Desktop\dongmin\3d_printer_vib_rms_normal"
selected_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
file_names = [f"3d_printer_vib_rms_normal_{i}.csv" for i in selected_indices]

# 모든 RMS 데이터를 리스트에 저장
all_rms = []
for file in file_names:
    file_path = os.path.join(base_path, file)
    df = pd.read_csv(file_path, encoding='utf-8')
    all_rms.append(df['RMS'].values)

# 시퀀스 길이 맞추기 (가장 짧은 길이에 맞춤)
min_len = min(len(rms) for rms in all_rms)
all_rms_aligned = [rms[:min_len] for rms in all_rms]

# 데이터 정규화 (개별 파일별로 정규화)
scaler = MinMaxScaler()
all_rms_scaled = []
for rms in all_rms_aligned:
    scaled = scaler.fit_transform(rms.reshape(-1, 1)).flatten()
    all_rms_scaled.append(scaled)
all_rms_array = np.array(all_rms_scaled)  # shape: (30, min_len)

# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    sequences = []
    for rms in data:
        for i in range(len(rms) - seq_length + 1):
            seq = rms[i:i+seq_length]
            sequences.append(seq.reshape(-1, 1))  # (seq_length, 1)
    return np.array(sequences)

SEQ_LENGTH = 30
X = create_sequences(all_rms_array, SEQ_LENGTH)
X_train_full, X_test = train_test_split(X, test_size=0.2, shuffle=False)
# DQN을 위한 validation set 분리
X_train, X_val = train_test_split(X_train_full, test_size=0.2, shuffle=False)

# 노이즈 추가 함수 정의
def add_noise(data, noise_factor=0.05):
    noise = np.random.normal(loc=0, scale=noise_factor, size=data.shape)
    noisy_data = data + noise
    return np.clip(noisy_data, 0.0, 1.0)

# 노이즈 추가된 학습 데이터 생성
X_train_noisy = add_noise(X_train)
X_val_noisy = add_noise(X_val)

# LSTM-VAE 모델 정의
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder (입력 shape: (SEQ_LENGTH, 1))
encoder_inputs = layers.Input(shape=(SEQ_LENGTH, 1))
x = layers.LSTM(64, return_sequences=True)(encoder_inputs)
x = layers.LSTM(32)(x)
z_mean = layers.Dense(8)(x)
z_log_var = layers.Dense(8)(x)
z = Sampling()([z_mean, z_log_var])
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = layers.Input(shape=(8,))
x = layers.RepeatVector(SEQ_LENGTH)(latent_inputs)
x = layers.LSTM(32, return_sequences=True)(x)
x = layers.LSTM(64, return_sequences=True)(x)
decoder_outputs = layers.TimeDistributed(layers.Dense(1))(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")

# VAE 구성
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = Model(encoder_inputs, vae_outputs, name="vae")

vae.encoder = encoder
vae.decoder = decoder

# Loss 함수
reconstruction_loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(encoder_inputs - vae_outputs), axis=[1,2])
)
kl_loss = -0.5 * tf.reduce_mean(
    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
)
vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)

# 학습 곡선 저장용 Callback
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

# 배치 단위 손실 기록
class BatchLossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

batch_loss_history = BatchLossHistory()
loss_history = LossHistory()

# VAE 컴파일 및 학습
vae.compile(optimizer='adam')
vae.fit(
    X_train_noisy, X_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val_noisy, X_val),
    callbacks=[loss_history, batch_loss_history, tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# 학습 곡선 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history.losses, label='Train Loss')
plt.plot(loss_history.val_losses, label='Validation Loss')
plt.title('Epoch Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(batch_loss_history.batch_losses)
plt.title('Batch Loss')
plt.tight_layout()
plt.show()


# 이상 데이터 주입
def inject_anomalies(data, ratio=0.07, min_strength=3.0, max_strength=6.0):
    data = data.copy()
    n_anomalies = int(len(data) * ratio)
    anomaly_indices = []
    seq_len = data.shape[1]
    curve_types = ['spike'] * 6 + ['step', 'linear', 'exp', 'gradual_increase']
    used_indices = set()
    while len(anomaly_indices) < n_anomalies:
        idx = np.random.randint(0, len(data))
        # 인접 이상치 방지 (한 시퀀스에 이상치가 몰리지 않게)
        if any(abs(idx - uidx) < 5 for uidx in used_indices):
            continue
        used_indices.add(idx)
        anomaly_indices.append(idx)
        strength = np.random.uniform(min_strength, max_strength)
        drift_magnitude = strength * np.std(data)
        curve_type = np.random.choice(curve_types)
        if curve_type == 'spike':
            spike_start = np.random.randint(seq_len//4, 3*seq_len//4)
            spike_width = np.random.randint(1, 3)
            drift = np.zeros(seq_len)
            drift[spike_start:spike_start+spike_width] = drift_magnitude * np.random.uniform(3.0, 5.0)
        elif curve_type == 'step':
            step_point = np.random.randint(seq_len//4, 3*seq_len//4)
            drift = np.concatenate([
                np.zeros(step_point),
                np.full(seq_len-step_point, drift_magnitude)
            ])
        elif curve_type == 'linear':
            drift = np.linspace(0, drift_magnitude, seq_len)
        elif curve_type == 'exp':
            x = np.linspace(0, 1, seq_len)
            drift = drift_magnitude * (np.exp(3*x) - 1) / (np.exp(3) - 1)
        elif curve_type == 'gradual_increase':
            start_point = np.random.randint(seq_len//3)
            drift = np.zeros(seq_len)
            drift[start_point:] = np.linspace(0, drift_magnitude, seq_len-start_point)
        data[idx] += drift.reshape(-1, 1)
    return data, anomaly_indices

X_val_anomalous, val_anomaly_idx = inject_anomalies(X_val)
X_test_anomalous, test_anomaly_idx = inject_anomalies(X_test)

# 강화학습 환경 정의
class DynamicThresholdEnv(Env):
    def __init__(self, model, data, anomaly_indices):
        super().__init__()
        self.model = model
        self.data = data
        self.anomaly_indices = anomaly_indices
        self.current_step = 0
        self.action_space = Discrete(41) # 0~40
        self.observation_space = Box(low=0.0, high=1.0, shape=(23,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.recent_errors = [0.0, 0.0, 0.0]
        self.recent_pred_errors = []
        self.current_threshold = 0.15
        self.prev_threshold = 0.15
        self.fn_streak = 0
        self.fp_streak = 0
        self.recent_predictions = []
        self.error_history = []
        self.tn_count = 0
        self.tp_count = 0
        self.fp_count = 0
        self.fn_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        window = self.data[self.current_step]
        z_mean, _, _ = self.model.encoder.predict(window[np.newaxis], verbose=0)
        recon = self.model.predict(window[np.newaxis], verbose=0).squeeze(0)
        error = np.mean(np.abs(window - recon))
        error_vec = np.mean(np.abs(window - recon), axis=0)
        error = np.clip(error, 0, 1)
        error_vec = np.clip(error_vec, 0, 1)

        self.error_history.append(error)
        if len(self.error_history) > 10:
            self.error_history.pop(0)

        error_mean = np.mean(self.error_history)
        error_std = np.std(self.error_history)
        error_trend = self.error_history[-1] - self.error_history[0] if len(self.error_history) > 1 else 0

        self.recent_errors = self.recent_errors[1:] + [error]
        error_diff = self.recent_errors[-1] - self.recent_errors[-2]
        error_diff = np.clip(error_diff, -1, 1) * 0.5 + 0.5

        min_err = np.min(self.error_history)
        max_err = np.max(self.error_history)
        median_err = np.median(self.error_history)
        last3 = self.error_history[-3:] if len(self.error_history) >= 3 else [0,0,0]

        # 최근 FP/FN 비율
        if len(self.recent_predictions) > 0:
            recent_fp_rate = sum(1 for p in self.recent_predictions[-10:] if p == 'FP') / min(10, len(self.recent_predictions))
            recent_fn_rate = sum(1 for p in self.recent_predictions[-10:] if p == 'FN') / min(10, len(self.recent_predictions))
        else:
            recent_fp_rate = 0.0
            recent_fn_rate = 0.0

        obs = np.concatenate([
            z_mean[0].flatten()[:8],
            np.array([
                self.recent_errors[-2], self.recent_errors[-1], error_diff,
                error_mean, error_std, error_trend,
                min_err, max_err, median_err,
                *last3,
                self.current_threshold / 0.5,
                recent_fp_rate, recent_fn_rate
            ])
        ])

        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        self.current_threshold = 0.02 + action * 0.003
        self.current_threshold = np.clip(self.current_threshold, 0.02, 0.15)

        window = self.data[self.current_step]
        recon = self.model.predict(window[np.newaxis], verbose=0)
        error = np.mean(np.abs(window - recon))

        is_anomaly = self.current_step in self.anomaly_indices
        y_true = 1 if is_anomaly else 0
        y_pred = 1 if error > self.current_threshold else 0

        # 축소된 보상 구조
        if y_true == 1 and y_pred == 1:  # TP
            reward = 1.0
            self.tp_count += 1
            self.fn_streak = 0
            self.fp_streak = 0
            prediction_type = 'TP'
        elif y_true == 1 and y_pred == 0:  # FN
            reward = -0.8
            self.fn_count += 1
            self.fn_streak += 1
            self.fp_streak = 0
            prediction_type = 'FN'
        elif y_true == 0 and y_pred == 1:  # FP
            reward = -0.3
            self.fp_count += 1
            self.fp_streak += 1
            self.fn_streak = 0
            prediction_type = 'FP'
        else:  # TN
            reward = 0.2
            self.tn_count += 1
            self.fn_streak = 0
            self.fp_streak = 0
            prediction_type = 'TN'

        self.recent_predictions.append(prediction_type)
        if len(self.recent_predictions) > 20:
            self.recent_predictions.pop(0)

        # 연속 FN 패널티 (축소)
        if self.fn_streak > 2:
            reward -= 0.05 * self.fn_streak

        # 연속 FP 패널티 (축소)
        if self.fp_streak > 3:
            reward -= 0.05 * (self.fp_streak - 3)

        # Precision 낮을 때 패널티
        recent_precision = self.tp_count / (self.tp_count + self.fp_count) if (self.tp_count + self.fp_count) > 0 else 1.0
        if recent_precision < 0.6:
            reward -= 0.1

        # threshold 변화 비용 (축소)
        threshold_change_cost = abs(self.current_threshold - self.prev_threshold) * 0.02
        reward -= threshold_change_cost
        self.prev_threshold = self.current_threshold

        # 극단적 threshold 페널티 (유지)
        if self.current_threshold >= 0.30:
            reward -= 0.1
        elif self.current_threshold <= 0.05:
            reward -= 0.05

        if self.current_step % 100 == 0 or done:
            print(f"[Step {self.current_step}] TP={self.tp_count} FN={self.fn_count} FP={self.fp_count} TN={self.tn_count}")
            print(f"  Precision={recent_precision:.3f} FP_streak={self.fp_streak} FN_streak={self.fn_streak}")
            print(f"  Threshold={self.current_threshold:.3f} Reward={reward:.2f} Error={error:.4f}")

        return self._get_obs(), reward, done, False, {}

# DQN 학습 환경 및 파라미터
log_path = "./dqn_logs"
tb_name = "DQN_run1"

train_env = Monitor(DynamicThresholdEnv(vae, X_val_anomalous, val_anomaly_idx))
os.makedirs(log_path, exist_ok=True)
new_logger = configure(os.path.join(log_path, tb_name), ["stdout", "csv", "tensorboard"])

rl_model = DQN(
    "MlpPolicy",
    train_env,
    verbose=1,
    learning_rate=2e-4,
    buffer_size=150000,
    learning_starts=3000,
    exploration_fraction=0.6,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.2,
    target_update_interval=1500,
    gamma=0.98,
    train_freq=2,
    gradient_steps=2,
    tensorboard_log=log_path,
    policy_kwargs=dict(net_arch=[128])
)

rl_model.set_logger(new_logger)
rl_model.learn(total_timesteps=50000, tb_log_name=tb_name)

# 추론 함수
def dynamic_threshold_detection_fast(model, data):
    z_mean, _, _ = vae.encoder.predict(data, verbose=0)
    recon = vae.predict(data, verbose=0)
    errors = np.mean(np.abs(data - recon), axis=(1, 2))
    thresholds = []
    error_history = []
    
    for i in range(len(data)):
        error_history.append(errors[i])
        if len(error_history) > 10:
            error_history.pop(0)
        
        error_mean = np.mean(error_history)
        error_std = np.std(error_history)
        error_trend = error_history[-1] - error_history[0] if len(error_history) > 1 else 0
        
        if i >= 2:
            recent_errors = [errors[i-2], errors[i-1], errors[i]]
            error_diff = errors[i] - errors[i-1]
        elif i == 1:
            recent_errors = [0.0, errors[i-1], errors[i]]
            error_diff = errors[i] - errors[i-1]
        else:
            recent_errors = [0.0, 0.0, errors[i]]
            error_diff = 0.0
        
        error_diff = np.clip(error_diff, -1, 1) * 0.5 + 0.5
        min_err = np.min(error_history)
        max_err = np.max(error_history)
        median_err = np.median(error_history)
        last3 = error_history[-3:] if len(error_history) >= 3 else [0,0,0]
        recent_fp_rate = 0.0
        recent_fn_rate = 0.0
        current_threshold = thresholds[-1] if thresholds else 0.15
        
        obs = np.concatenate([
            z_mean[i].flatten()[:8],
            np.array([
                recent_errors[-2], recent_errors[-1], error_diff,
                error_mean, error_std, error_trend,
                min_err, max_err, median_err,
                *last3,
                current_threshold / 0.5,
                recent_fp_rate, recent_fn_rate
            ])
        ])
        
        action, _ = model.predict(obs[np.newaxis, :].astype(np.float32), deterministic=True)
        threshold = 0.08 + action[0] * 0.006
        thresholds.append(threshold)
    
    return np.array(thresholds)

# 평가 함수
def evaluate(model, X, true_anomalies):
    recon = vae.predict(X, verbose=0)
    errors = np.mean(np.abs(X - recon), axis=(1, 2))
    if isinstance(model, DQN):
        thresholds = dynamic_threshold_detection_fast(model, X)
        y_pred = (errors > thresholds).astype(int)
    else:
        threshold = np.percentile(errors, 95)
        y_pred = (errors > threshold).astype(int)
    y_true = np.zeros(len(X), dtype=int)
    y_true[true_anomalies] = 1
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"Accuracy : {accuracy:.4f}")
    return f1, precision, recall, accuracy

print("Denoising LSTM-VAE Evaluation:")
evaluate(vae, X_test_anomalous, test_anomaly_idx)
print("\nDenoising LSTM-VAE + DQN Evaluation:")
evaluate(rl_model, X_test_anomalous, test_anomaly_idx)