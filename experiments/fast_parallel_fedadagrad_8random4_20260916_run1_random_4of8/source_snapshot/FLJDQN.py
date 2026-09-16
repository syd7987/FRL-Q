# -*- coding: utf-8 -*-
"""
Modified: Remove Weight Loading in Worker Process (Debugging Mode)
"""

import cubic_spline_planner
import numpy as np
import random
import os
import json

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
_cpu_threads_per_worker = os.environ.get("FL_CPU_THREADS_PER_WORKER", "1")
os.environ.setdefault("OMP_NUM_THREADS", _cpu_threads_per_worker)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _cpu_threads_per_worker)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import mixed_precision
from collections import deque
import math
import sys
import pathlib
import pandas as pd
import gc
import copy
import pickle
import traceback

try:
    from experiments.heterogeneity.aggregation import cosine_aware_delta
    from experiments.heterogeneity.metrics import (
        normalized_std,
        td_error_mean,
        update_direction_metrics,
    )
except ImportError:
    cosine_aware_delta = None
    normalized_std = None
    td_error_mean = None
    update_direction_metrics = None

import multiprocessing
import time 
from tensorflow.keras import backend as K 

try:
    import config
except ImportError:
    import FLJDQN_config as config

gc.collect()


rho = config.rho
lamda = config.lamda
eps = config.eps
initial_eps = config.initial_eps

round_fre = getattr(config, "round_fre", 10)
checkpoint_every_rounds = max(1, int(getattr(config, "checkpoint_every_rounds", 1)))
eps_decay = config.eps_decay
batch_siz = config.batch_siz
D_h = config.D_h
n_episode = config.n_episode
timestep = config.timestep
seed = config.seed
t_up = config.t_up
dt = config.dt
target_vel = config.target_vel
num_agents = config.num_agents
route_mode = getattr(config, "route_mode", "train")
agent_route_assignments = getattr(config, "agent_route_assignments", list(range(num_agents)))
experiment_name = getattr(config, "experiment_name", "")
aggregation_method = getattr(config, "aggregation_method", "fedadagrad").lower()
window_average_enabled = getattr(config, "window_average_enabled", False)
window_average_size = max(1, int(getattr(config, "window_average_size", 5)))
server_eta = getattr(config, "server_eta", 0.1)
server_beta1 = getattr(config, "server_beta1", 0.9)
server_beta2 = getattr(config, "server_beta2", 0.99)
server_tau = getattr(config, "server_tau", 1e-6)
fedprox_mu = float(getattr(config, "fedprox_mu", 0.01))
heterogeneity_condition = getattr(config, "heterogeneity_condition", "")
heterogeneity_log_enabled = bool(getattr(config, "heterogeneity_log_enabled", False))
cosine_lambda = float(getattr(config, "cosine_lambda", 1.0))
client_selection_mode = (
    str(getattr(config, "client_selection_mode", "all"))
    .strip()
    .lower()
    .replace("-", "_")
)
client_selection_mode = {
    "standalone": "single",
    "single_vehicle": "single",
    "reward": "reward_high",
    "cosine": "cosine_high",
    "balanced_random": "count_balanced_random",
    "balanced_reward": "count_balanced_reward",
}.get(client_selection_mode, client_selection_mode)
client_selection_agent_id = max(0, min(int(getattr(config, "client_selection_agent_id", 0)), num_agents - 1))
client_selection_budget = max(1, min(int(getattr(config, "client_selection_budget", num_agents)), num_agents))
worker_device = str(getattr(config, "worker_device", "gpu")).lower()
mixed_precision_enabled = bool(getattr(config, "mixed_precision_enabled", True))
if client_selection_mode == "all":
    client_selection_budget = num_agents
elif client_selection_mode == "single":
    client_selection_budget = 1

if aggregation_method == "cosine_fedavg" and cosine_aware_delta is None:
    raise ImportError(
        "cosine_fedavg requires experiments.heterogeneity.aggregation, "
        "which is not included with the three supplied files."
    )
if heterogeneity_log_enabled and update_direction_metrics is None:
    raise ImportError(
        "FL_HETEROGENEITY_LOG requires experiments.heterogeneity.metrics, "
        "which is not included with the three supplied files."
    )

METRIC_COLUMNS = [
    "episode",
    "path_id",
    "avg_lateral_error",
    "max_lateral_error",
    "total_reward",
    "avg_reward",
]

HETEROGENEITY_AGENT_COLUMNS = [
    "round",
    "seed",
    "condition",
    "algorithm",
    "agent_id",
    "path_id",
    "vehicle_id",
    "update_norm",
    "cosine_to_mean",
    "td_error_mean",
    "avg_lateral_error",
    "max_lateral_error",
    "total_reward",
    "avg_reward",
    "episode_length",
    "done_reason",
]

HETEROGENEITY_ROUND_COLUMNS = [
    "round",
    "seed",
    "condition",
    "algorithm",
    "H_cos",
    "H_TD",
    "mean_update_norm",
    "std_update_norm",
    "global_avg_lateral_error",
    "global_max_lateral_error",
    "global_avg_reward",
    "global_total_reward",
    "performance_gap",
]

CLIENT_SELECTION_AGENT_COLUMNS = [
    "round",
    "seed",
    "condition",
    "algorithm",
    "selection_mode",
    "selection_budget",
    "agent_id",
    "selection_score",
    "selected_for_aggregation",
]

CLIENT_SELECTION_ROUND_COLUMNS = [
    "round",
    "seed",
    "condition",
    "algorithm",
    "selection_mode",
    "selection_budget",
    "candidate_agent_ids",
    "selected_agent_ids",
]


Acceleration = config.Acceleration
Delta = config.Delta
action_space = []
for acc in Acceleration:
    for delta in Delta:
        action_space.append((acc, delta))
action_len = len(action_space)


def set_global_seed(base_seed, offset=0, context="Main"):
    run_seed = int(base_seed) + int(offset)
    os.environ["PYTHONHASHSEED"] = str(run_seed)
    random.seed(run_seed)
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)
    try:
        tf.keras.utils.set_random_seed(run_seed)
    except AttributeError:
        pass
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    print(f"[{context}] Random seed fixed: {run_seed}")


class States:
    def __init__(self):
        self.x = []
        self.y = []
    def append(self, state):
        self.x.append(state[0])
        self.y.append(state[1])

class NonLinearBicycleModel():
    def __init__(self, L, Lf, Lr, m, Cf, Cr, Iz, max_steer, noise_Var, noise_Var_xy, x=0.0, y=0.0, yaw=0.0, vx=10/3.6, vy=0, omega=0.0):
        self.L = L
        self.Lf = Lf
        self.Lr = Lr
        self.m = m
        self.Cf = Cf
        self.Cr = Cr
        self.Iz = Iz
        self.max_steer = max_steer
        self.noiseVar = noise_Var
        self.noise_Var_xy = noise_Var_xy 
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.front_x = self.x + ((self.L/2) * math.cos(self.yaw))
        self.front_y = self.y + ((self.L/2) * math.sin(self.yaw))

    def update(self, throttle, delta):
        self.error_x = np.random.normal(loc=0, scale=math.sqrt(self.noise_Var_xy))
        self.error_y = np.random.normal(loc=0, scale=math.sqrt(self.noise_Var_xy))
        
        self.error_theta = np.random.normal(loc=0, scale=math.sqrt(self.noiseVar))
        
        delta = np.clip(delta, -self.max_steer, self.max_steer)
        self.x = self.x + self.vx * math.cos(self.yaw) * dt - self.vy * math.sin(self.yaw) * dt + self.error_x
        self.y = self.y + self.vx * math.sin(self.yaw) * dt + self.vy * math.cos(self.yaw) * dt + self.error_y
        self.yaw = self.yaw + self.omega * dt + self.error_theta
        self.yaw = normalize_angle_st(self.yaw)
        self.front_x = self.x + ((self.L/2) * math.cos(self.yaw))
        self.front_y = self.y + ((self.L/2) * math.sin(self.yaw))
        if self.vx != 0:
            Ffy = self.Cf * (delta - math.atan((self.vy + self.Lf * self.omega) / self.vx))
            Fry = self.Cr * (-math.atan((self.vy - self.Lr * self.omega) / self.vx))
        else:
            Ffy = 0
            Fry = 0
        self.vx = limit_to_range(self.vx + (throttle + self.vy * self.omega) * dt, 10/3.6, target_vel)
        self.vy = self.vy + (Fry / self.m + Ffy * math.cos(delta) / self.m - self.vx * self.omega) * dt
        self.omega = self.omega + (Ffy * self.Lf * math.cos(delta) - Fry * self.Lr) / self.Iz * dt

class Agent:
    def __init__(self, agent_id, cx_list, cy_list, cyaw_list):
        self.agent_id = agent_id
        self.state = []
        self.total_lateral_error = 0
        self.total_velocity = 0
        self.done = False
        self.lateral_errors = []
        self.epsilon = 1.0
        self.target_n = 0
        self.distance = 0
        vehicle_params = config.vehicles[agent_id]
        
        self.state_dqn = NonLinearBicycleModel(
            L=vehicle_params.L,
            Lf=vehicle_params.Lf,
            Lr=vehicle_params.Lr,
            m=vehicle_params.m,
            Cf=vehicle_params.Cf,
            Cr=vehicle_params.Cr,
            Iz=vehicle_params.Iz,
            max_steer=vehicle_params.max_steer,
            noise_Var=vehicle_params.noise_Var,
            noise_Var_xy=vehicle_params.noise_Var_xy,
            x=0.0, y=0.0, yaw=cyaw_list[0], vx=10/3.6, vy=0, omega=0.0
        )
        
        self.state = [self.state_dqn.x, self.state_dqn.y, self.state_dqn.yaw]
        n_state_diff_low = tran_diff_dis(self.state_dqn, cx_list, cy_list, cyaw_list)
        self.state_diff_l = n_state_diff_low

def normalize_angle_st(angle):
    while angle > np.pi:
        angle -= 2.0*np.pi
    while angle < -np.pi:
        angle += 2.0*np.pi
    return angle

def limit_to_range(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    return value

def car_arr_dis(n_state, Cx, Cy):
    rx = n_state[0] - Cx[-1]
    ry = n_state[1] - Cy[-1]
    arr_dis = np.hypot(rx, ry)
    return arr_dis

def tran_diff_dis(state_main, Cx, Cy, Cyaw):
    rx = np.array(Cx)-state_main.front_x
    ry = np.array(Cy)-state_main.front_y
    d = np.hypot(rx, ry)
    ind_f = np.argmin(d)
    xe=np.cos(state_main.yaw)*rx[ind_f]+np.sin(state_main.yaw)*ry[ind_f]
    ye=-np.sin(state_main.yaw)*rx[ind_f]+np.cos(state_main.yaw)*ry[ind_f]
    theta_e=normalize_angle_st(Cyaw[ind_f]-state_main.yaw)
    n_state_diff_low = [xe, ye, theta_e, state_main.omega, state_main.vx/target_vel, state_main.vy]
    return n_state_diff_low

def RL_step_high(action_index, State_main, cx, cy, cyaw, index, timestep):
    throttle, delta = action_space[action_index]
    State_main.update(throttle, delta)
    new_state = [State_main.x, State_main.y, State_main.yaw]
    new_state_front = [State_main.front_x, State_main.front_y, State_main.yaw]
    
    rx = np.array(cx) - new_state[0]
    ry = np.array(cy) - new_state[1]
    d = np.hypot(rx, ry)
    ind = np.argmin(d)
    lateral_error = d[ind]
    
    orientation_error = normalize_angle_st(new_state[2] - cyaw[ind])
    
    reward = -lateral_error + State_main.vx / target_vel * dt
    arr_dis = car_arr_dis(new_state, cx, cy)
    done = (index == timestep - 1) or arr_dis < target_vel * dt
    
    rx_f = np.array(cx) - new_state_front[0]
    ry_f = np.array(cy) - new_state_front[1]
    d_front = np.hypot(rx_f, ry_f)
    ind_f = np.argmin(d_front)
    xe = np.cos(new_state_front[2]) * rx_f[ind_f] + np.sin(new_state_front[2]) * ry_f[ind_f]
    ye = -np.sin(new_state_front[2]) * rx_f[ind_f] + np.cos(new_state_front[2]) * ry_f[ind_f]
    theta_e = normalize_angle_st(cyaw[ind_f] - new_state_front[2])
    n_state_diff_high = [xe, ye, theta_e, State_main.omega, State_main.vx / target_vel, State_main.vy]

    return n_state_diff_high, reward, done, lateral_error, orientation_error, new_state

def configure_tensorflow_runtime(process_label):
    """Configure the accelerator before the process creates any TF variables."""
    gpus = tf.config.list_physical_devices('GPU')
    use_gpu = worker_device == "gpu" and bool(gpus)

    if use_gpu:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        if mixed_precision_enabled:
            mixed_precision.set_global_policy("mixed_float16")
        device_name = "GPU"
    else:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError:
            pass
        mixed_precision.set_global_policy("float32")
        device_name = "CPU"

    print(
        f"[{process_label}] TensorFlow device={device_name}, "
        f"mixed_precision={mixed_precision.global_policy().name}"
    )
    return use_gpu


def deep_network(state_dim, action_dim):
    mlp = Sequential()
    mlp.add(Dense(128, input_dim=state_dim, activation='relu'))
    mlp.add(Dense(128, activation='relu'))
    # Keep Q values and the loss in float32 when hidden layers use float16.
    mlp.add(Dense(action_dim, activation='linear', dtype='float32'))
    mlp.compile(loss='mse', optimizer='Adam')
    return mlp

def build_dqn_training_batch(D, Model, Model_target):
    mini_batch = np.asarray(random.sample(D, batch_siz), dtype=object)
    state_diff = np.asarray([mini_batch[i, 0] for i in range(batch_siz)], dtype=np.float32)
    action = mini_batch[:, 1].astype(np.int32)
    reward = mini_batch[:, 2].astype(np.float32)
    state1_diff = np.asarray([mini_batch[i, 3] for i in range(batch_siz)], dtype=np.float32)
    done = mini_batch[:, 4].astype(bool)

    target = Model(state_diff, training=False).numpy()
    target1 = Model_target(state1_diff, training=False).numpy()

    rows = np.arange(batch_siz)
    current_action_q = target[rows, action]
    bootstrapped_q = reward + lamda * np.max(target1, axis=1)
    target[rows, action] = np.where(
        done,
        reward,
        (1 - rho) * current_action_q + rho * bootstrapped_q,
    )

    return state_diff, action, target.astype(np.float32)

def model_learning(D, Model, Model_target):
    if len(D) < batch_siz:
        return

    state_diff, _, target = build_dqn_training_batch(D, Model, Model_target)
    Model.train_on_batch(state_diff, target)

def model_learning_fedprox(D, Model, Model_target, proximal_weights, proximal_mu):
    if len(D) < batch_siz:
        return

    if proximal_weights is None or proximal_mu <= 0:
        model_learning(D, Model, Model_target)
        return

    state_diff, action, target = build_dqn_training_batch(D, Model, Model_target)
    proximal_tensors = [
        tf.convert_to_tensor(global_w, dtype=variable.dtype)
        for variable, global_w in zip(Model.trainable_variables, proximal_weights)
    ]

    with tf.GradientTape() as tape:
        q_values = Model(state_diff, training=True)
        action_mask = tf.one_hot(action, action_len, dtype=q_values.dtype)
        predicted_action_q = tf.reduce_sum(q_values * action_mask, axis=1)
        target_action_q = tf.reduce_sum(tf.convert_to_tensor(target, dtype=q_values.dtype) * action_mask, axis=1)
        dqn_loss = tf.reduce_mean(tf.square(target_action_q - predicted_action_q))
        proximal_loss = tf.add_n([
            tf.reduce_sum(tf.square(variable - global_w))
            for variable, global_w in zip(Model.trainable_variables, proximal_tensors)
        ])
        total_loss = dqn_loss + (proximal_mu / 2.0) * proximal_loss

    gradients = tape.gradient(total_loss, Model.trainable_variables)
    Model.optimizer.apply_gradients(zip(gradients, Model.trainable_variables))

def worker_process(agent_id, start_episode, episodes_to_run, global_weight_path, cx_list, cy_list, cyaw_list, temp_dir):
    
    try:
        use_gpu = configure_tensorflow_runtime(f"Worker {agent_id}")
        worker_seed_offset = (agent_id + 1) * 100000 + int(start_episode)
        set_global_seed(seed, worker_seed_offset, context=f"Worker {agent_id}")

        print(f"[Worker {agent_id}] started on {'GPU' if use_gpu else 'CPU'}.")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)

        local_model = deep_network(6, action_len)
        local_target_model = deep_network(6, action_len)

        @tf.function(reduce_retracing=True)
        def infer_q(state_tensor):
            return local_model(state_tensor, training=False)
        local_target_weight_path = os.path.join(temp_dir, f"local_target_weights_{agent_id}.weights.h5")
        proximal_weights = None
        
        try:
             local_model.load_weights(global_weight_path)
             if client_selection_mode == "single" and os.path.exists(local_target_weight_path):
                 local_target_model.load_weights(local_target_weight_path)
             else:
                 local_target_model.load_weights(global_weight_path)
             if aggregation_method == "fedprox":
                 proximal_weights = [w.copy() for w in local_model.get_weights()]
        except Exception as e:
             print(f"[Worker {agent_id}]: {e}")
            
        buffer_path = os.path.join(temp_dir, f"replay_buffer_{agent_id}.pkl")
        if os.path.exists(buffer_path):
            try:
                with open(buffer_path, 'rb') as f:
                    D_local = pickle.load(f)
            except Exception as e:
                print(f"[Worker {agent_id}] {e}")
                D_local = deque(maxlen=3000)
        else:
            D_local = deque(maxlen=3000)
        
        cx_worker = cx_list[agent_id]
        cy_worker = cy_list[agent_id]
        cyaw_worker = cyaw_list[agent_id]
        
        worker_eps = eps * (eps_decay ** start_episode)
        
        worker_results = [] 
        round_rewards = []
        re_target_n_worker = 0

        for i in range(start_episode, start_episode + episodes_to_run):
            
            agent = Agent(agent_id, cx_worker, cy_worker, cyaw_worker)
            agent.target_n = re_target_n_worker 
            worker_eps *= eps_decay
            if worker_eps < 0.01: worker_eps = 0.01

            Done = False
            episode_reward = 0.0
            
            for e in range(timestep):
                
                r = np.random.random()
                if r < worker_eps:
                    a_del = np.random.randint(0, action_len)
                else:
                    state_in = np.asarray(agent.state_diff_l, dtype=np.float32).reshape(1, 6)
                    q_h = infer_q(tf.convert_to_tensor(state_in))
                    a_del = int(tf.argmax(q_h[0]).numpy())
                
                new_state_diff_l, re_low, done, lateral_error, orientation_error, new_state = RL_step_high(a_del, agent.state_dqn, cx_worker, cy_worker, cyaw_worker, e, timestep)
                episode_reward += float(re_low)
                
                D_local.append((agent.state_diff_l, a_del, re_low, new_state_diff_l, done))
                
                if len(D_local) > batch_siz:
                    if aggregation_method == "fedprox":
                        model_learning_fedprox(
                            D_local,
                            local_model,
                            local_target_model,
                            proximal_weights=proximal_weights,
                            proximal_mu=fedprox_mu,
                        )
                    else:
                        model_learning(D_local, local_model, local_target_model)
                
                agent.state_diff_l = new_state_diff_l
                agent.state = new_state
                
                agent.total_velocity += agent.state_dqn.vx
                agent.total_lateral_error += lateral_error
                agent.lateral_errors.append(lateral_error)
                
                agent.target_n += 1
                if agent.target_n % t_up == 0:
                    local_target_model.set_weights(local_model.get_weights())
                
                Done = done.item() if isinstance(done, np.ndarray) else done
                
                if Done:
                    break 
            
            re_target_n_worker = agent.target_n 
            round_rewards.append(episode_reward)

            num_steps = len(agent.lateral_errors)
            if num_steps > 0:
                avg_lateral_error = agent.total_lateral_error / num_steps
                max_lateral_error = max(agent.lateral_errors)
            else:
                avg_lateral_error = 0
                max_lateral_error = 0

            worker_results.append({
                'episode': i,
                'path_id': agent_id,
                'avg_lateral_error': avg_lateral_error,
                'max_lateral_error': max_lateral_error,
                'total_reward': episode_reward,
                'avg_reward': episode_reward / max(num_steps, 1),
                'episode_length': num_steps,
                'done_reason': 'done' if Done else 'timestep_limit',
            })
            
            if (i + 1) % 1 == 0:
                print(f"  [Worker {agent_id}] episode {i} finished. (AvgLatErr: {avg_lateral_error:.4f}, Steps: {num_steps})")

        # --- 모든 episode 완료 후 결과 저장 ---
        
        # 1. local model weight 저장
        local_weight_path = os.path.join(temp_dir, f"local_weights_{agent_id}.weights.h5")
        local_model.save_weights(local_weight_path)
        local_target_model.save_weights(local_target_weight_path)
        
        # 2. worker metric CSV 저장
        temp_csv_path = os.path.join(temp_dir, f"temp_results_{agent_id}.csv")
        if worker_results:
            pd.DataFrame(worker_results).to_csv(temp_csv_path, index=False)

        # Replay buffer 저장
        try:
            with open(buffer_path, 'wb') as f:
                pickle.dump(D_local, f)
        except Exception as e:
            print(f"[Worker {agent_id}]: {e}")

        # 3. 메모리 정리
        del local_model
        del local_target_model
        K.clear_session()
        gc.collect()
        print(f"[Worker {agent_id}].")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Worker {agent_id}")
        print(traceback.format_exc())
        print("-" * 60)


def build_agent_route_assignments(num_agents_value, base_assignments):
    if not base_assignments:
        raise ValueError("agent_route_assignments must not be empty")

    if len(base_assignments) == num_agents_value:
        return list(base_assignments)

    print(f"[Path Config] agent_route_assignments length ({len(base_assignments)}) != num_agents ({num_agents_value}). Cycling assignments.")
    return [base_assignments[i % len(base_assignments)] for i in range(num_agents_value)]


def load_direct_paths(mode="train", num_agents_override=None, agent_route_assignments_override=None):
    train_paths = config.train_paths
    test_path = config.test_path

    cx_list, cy_list, cyaw_list = [], [], []
    local_num_agents = num_agents if num_agents_override is None else num_agents_override

    if mode == "train":
        route_assignments_raw = agent_route_assignments if agent_route_assignments_override is None else agent_route_assignments_override
        route_assignments = build_agent_route_assignments(local_num_agents, route_assignments_raw)

        for agent_idx in range(local_num_agents):
            route_idx = route_assignments[agent_idx]
            route_idx = int(route_idx) % len(train_paths)
            path = train_paths[route_idx]
            cx, cy, cyaw = cubic_spline_planner.calc_spline_course(path['AX'], path['AY'], ds=0.1)
            cx_list.append(list(cx))
            cy_list.append(list(cy))
            cyaw_list.append(list(cyaw))
    else:
        cx, cy, cyaw = cubic_spline_planner.calc_spline_course(test_path['AX'], test_path['AY'], ds=0.1)
        for _ in range(local_num_agents):
            cx_list.append(list(cx))
            cy_list.append(list(cy))
            cyaw_list.append(list(cyaw))

    return cx_list, cy_list, cyaw_list

def ensure_metric_csv_header(csv_path):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        pd.DataFrame(columns=METRIC_COLUMNS).to_csv(csv_path, index=False)
        return

    try:
        existing_header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"[Error] CSV header check failed: {e}")
        return

    if existing_header == METRIC_COLUMNS:
        return

    try:
        repaired_df = pd.read_csv(csv_path, names=METRIC_COLUMNS, header=0)
        repaired_df.to_csv(csv_path, index=False)
        print(f"[Main] Repaired metric CSV header: {csv_path}")
    except Exception as e:
        print(f"[Error] CSV header repair failed: {e}")

def average_client_deltas(client_deltas, old_global_weights):
    num_clients = len(client_deltas)
    avg_delta_t = [np.zeros_like(w) for w in old_global_weights]
    for delta_i in client_deltas:
        for k in range(len(avg_delta_t)):
            avg_delta_t[k] += delta_i[k]
    return [d / num_clients for d in avg_delta_t]

def apply_aggregation(old_global_weights, avg_delta_t, server_m, server_v):
    if aggregation_method in ("fedavg", "fedprox"):
        new_global_weights = [
            old_w + delta for old_w, delta in zip(old_global_weights, avg_delta_t)
        ]
        return new_global_weights, server_m, server_v

    if aggregation_method == "fedadam":
        server_m = [
            server_beta1 * m + (1 - server_beta1) * delta
            for m, delta in zip(server_m, avg_delta_t)
        ]

        server_v = [
            server_beta2 * v + (1 - server_beta2) * (delta ** 2)
            for v, delta in zip(server_v, avg_delta_t)
        ]

        new_global_weights = []
        for k in range(len(old_global_weights)):
            update_step = server_eta * server_m[k] / (np.sqrt(server_v[k]) + server_tau)
            new_global_weights.append(old_global_weights[k] + update_step)

        return new_global_weights, server_m, server_v

    if aggregation_method == "fedadagrad":
        server_v = [
            v + (delta ** 2)
            for v, delta in zip(server_v, avg_delta_t)
        ]

        new_global_weights = []
        for k in range(len(old_global_weights)):
            update_step = server_eta * avg_delta_t[k] / (np.sqrt(server_v[k]) + server_tau)
            new_global_weights.append(old_global_weights[k] + update_step)

        return new_global_weights, server_m, server_v

    raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

def append_heterogeneity_logs(
    csv_dir,
    round_index,
    selected_agent_ids,
    client_deltas,
    old_global_weights,
    all_round_results,
    temp_dir,
):
    if not heterogeneity_log_enabled or not client_deltas:
        return

    direction = update_direction_metrics(client_deltas, old_global_weights)
    td_errors = []
    agent_rows = []

    if all_round_results:
        perf_df = pd.concat(all_round_results, ignore_index=True)
        latest_perf = perf_df.sort_values("episode").groupby("path_id", as_index=False).tail(1)
    else:
        latest_perf = pd.DataFrame()

    for idx, agent_id in enumerate(selected_agent_ids):
        local_weight_path = os.path.join(temp_dir, f"local_weights_{agent_id}.weights.h5")
        local_target_weight_path = os.path.join(temp_dir, f"local_target_weights_{agent_id}.weights.h5")
        buffer_path = os.path.join(temp_dir, f"replay_buffer_{agent_id}.pkl")
        td_mean = 0.0

        try:
            if os.path.exists(buffer_path) and os.path.exists(local_weight_path):
                td_model = deep_network(6, action_len)
                td_target_model = deep_network(6, action_len)
                td_model.load_weights(local_weight_path)
                if os.path.exists(local_target_weight_path):
                    td_target_model.load_weights(local_target_weight_path)
                else:
                    td_target_model.load_weights(local_weight_path)
                with open(buffer_path, "rb") as f_buffer:
                    replay_buffer = pickle.load(f_buffer)
                td_mean = td_error_mean(replay_buffer, td_model, td_target_model, lamda)
                del td_model, td_target_model
        except Exception as e:
            print(f"[Main] TD error metric failed for agent {agent_id}: {e}")

        td_errors.append(td_mean)
        if not latest_perf.empty:
            perf_match = latest_perf[latest_perf["path_id"] == agent_id]
        else:
            perf_match = pd.DataFrame()

        if perf_match.empty:
            perf = {}
        else:
            perf = perf_match.iloc[0].to_dict()

        route_assignments = build_agent_route_assignments(num_agents, agent_route_assignments)
        path_id = int(route_assignments[agent_id])
        vehicle_id = 0 if getattr(config, "vehicle_mode", "combined") == "homogeneous" else int(agent_id % len(config.vehicle_parameters))

        agent_rows.append({
            "round": round_index,
            "seed": seed,
            "condition": heterogeneity_condition or "standard",
            "algorithm": aggregation_method,
            "agent_id": agent_id,
            "path_id": path_id,
            "vehicle_id": vehicle_id,
            "update_norm": direction["update_norm"][idx],
            "cosine_to_mean": direction["cosine_to_mean"][idx],
            "td_error_mean": td_mean,
            "avg_lateral_error": perf.get("avg_lateral_error", 0.0),
            "max_lateral_error": perf.get("max_lateral_error", 0.0),
            "total_reward": perf.get("total_reward", 0.0),
            "avg_reward": perf.get("avg_reward", 0.0),
            "episode_length": perf.get("episode_length", 0),
            "done_reason": perf.get("done_reason", ""),
        })

    h_td = normalized_std(td_errors)
    if all_round_results:
        perf_all = pd.concat(all_round_results, ignore_index=True)
        latest_perf_all = perf_all.sort_values("episode").groupby("path_id", as_index=False).tail(1)
        avg_lat_values = latest_perf_all["avg_lateral_error"].astype(float)
        round_row = {
            "round": round_index,
            "seed": seed,
            "condition": heterogeneity_condition or "standard",
            "algorithm": aggregation_method,
            "H_cos": direction["H_cos"],
            "H_TD": h_td,
            "mean_update_norm": direction["mean_update_norm"],
            "std_update_norm": direction["std_update_norm"],
            "global_avg_lateral_error": float(latest_perf_all["avg_lateral_error"].mean()),
            "global_max_lateral_error": float(latest_perf_all["max_lateral_error"].max()),
            "global_avg_reward": float(latest_perf_all["avg_reward"].mean()),
            "global_total_reward": float(latest_perf_all["total_reward"].mean()),
            "performance_gap": float(avg_lat_values.max() - avg_lat_values.min()),
        }
    else:
        round_row = {
            "round": round_index,
            "seed": seed,
            "condition": heterogeneity_condition or "standard",
            "algorithm": aggregation_method,
            "H_cos": direction["H_cos"],
            "H_TD": h_td,
            "mean_update_norm": direction["mean_update_norm"],
            "std_update_norm": direction["std_update_norm"],
            "global_avg_lateral_error": 0.0,
            "global_max_lateral_error": 0.0,
            "global_avg_reward": 0.0,
            "global_total_reward": 0.0,
            "performance_gap": 0.0,
        }

    agent_csv_path = os.path.join(csv_dir, "agent_level.csv")
    round_csv_path = os.path.join(csv_dir, "round_level.csv")
    pd.DataFrame(agent_rows, columns=HETEROGENEITY_AGENT_COLUMNS).to_csv(
        agent_csv_path,
        mode="a",
        header=not os.path.exists(agent_csv_path) or os.path.getsize(agent_csv_path) == 0,
        index=False,
    )
    pd.DataFrame([round_row], columns=HETEROGENEITY_ROUND_COLUMNS).to_csv(
        round_csv_path,
        mode="a",
        header=not os.path.exists(round_csv_path) or os.path.getsize(round_csv_path) == 0,
        index=False,
    )
    print(f"[Main] Heterogeneity metrics saved. (H_cos={direction['H_cos']:.4f}, H_TD={h_td:.4f})")

def append_window_model(window_models, weights, max_size):
    window_models.append([np.array(w, copy=True) for w in weights])
    if len(window_models) > max_size:
        del window_models[:-max_size]
    return window_models

def average_window_models(window_models):
    if not window_models:
        raise ValueError("window_models must not be empty")

    averaged = []
    for layer_idx in range(len(window_models[0])):
        layer_sum = np.zeros_like(window_models[0][layer_idx])
        for model_weights in window_models:
            layer_sum += model_weights[layer_idx]
        averaged.append(layer_sum / len(window_models))
    return averaged

def reset_replay_buffers(temp_dir, agent_ids):
    for agent_id in agent_ids:
        buffer_path = os.path.join(temp_dir, f"replay_buffer_{agent_id}.pkl")
        if os.path.exists(buffer_path):
            try:
                os.remove(buffer_path)
            except Exception as e:
                print(f"[Main] replay buffer reset failed for agent {agent_id}: {e}")

def load_selection_counts(count_path):
    counts = {agent_id: 0 for agent_id in range(num_agents)}
    if not os.path.exists(count_path):
        return counts

    try:
        with open(count_path, "r", encoding="utf-8") as f:
            saved_counts = json.load(f)
        for agent_id in range(num_agents):
            counts[agent_id] = int(saved_counts.get(str(agent_id), saved_counts.get(agent_id, 0)))
    except Exception as e:
        print(f"[Main] selection count load failed: {e}. Restart counts from zero.")
    return counts

def save_selection_counts(count_path, counts):
    try:
        with open(count_path, "w", encoding="utf-8") as f:
            json.dump({str(agent_id): int(counts.get(agent_id, 0)) for agent_id in range(num_agents)}, f, indent=2)
    except Exception as e:
        print(f"[Main] selection count save failed: {e}")

def update_selection_counts(counts, selected_agent_ids):
    for agent_id in selected_agent_ids:
        counts[agent_id] = int(counts.get(agent_id, 0)) + 1
    return counts

def latest_performance_by_agent(all_round_results):
    if not all_round_results:
        return {}

    perf_df = pd.concat(all_round_results, ignore_index=True)
    if perf_df.empty or "path_id" not in perf_df.columns:
        return {}

    latest_perf = perf_df.sort_values("episode").groupby("path_id", as_index=False).tail(1)
    return {
        int(row["path_id"]): row.to_dict()
        for _, row in latest_perf.iterrows()
    }

def select_aggregation_clients(
    selection_mode,
    selection_budget,
    successful_agent_ids,
    client_delta_by_agent_id,
    old_global_weights,
    all_round_results,
    round_index,
    selection_counts=None,
):
    candidate_agent_ids = [
        agent_id for agent_id in successful_agent_ids
        if agent_id in client_delta_by_agent_id
    ]
    if not candidate_agent_ids:
        return [], {}

    if selection_mode == "all":
        return list(candidate_agent_ids), {agent_id: 1.0 for agent_id in candidate_agent_ids}

    budget = min(max(1, int(selection_budget)), len(candidate_agent_ids))

    if selection_mode == "random":
        rng = random.Random(seed + (round_index + 1) * 1000003)
        selected_agent_ids = sorted(rng.sample(candidate_agent_ids, budget))
        scores = {
            agent_id: 1.0 if agent_id in selected_agent_ids else 0.0
            for agent_id in candidate_agent_ids
        }
        return selected_agent_ids, scores

    if selection_mode == "count_balanced_random":
        counts = selection_counts or {}
        rng = random.Random(seed + (round_index + 1) * 1000003)
        random_tiebreak = {agent_id: rng.random() for agent_id in candidate_agent_ids}
        ranked_agent_ids = sorted(
            candidate_agent_ids,
            key=lambda agent_id: (
                int(counts.get(agent_id, 0)),
                random_tiebreak[agent_id],
                agent_id,
            ),
        )
        selected_agent_ids = sorted(ranked_agent_ids[:budget])
        scores = {
            agent_id: 1.0 if agent_id in selected_agent_ids else 0.0
            for agent_id in candidate_agent_ids
        }
        return selected_agent_ids, scores

    if selection_mode in ("error_low", "reward_high", "count_balanced_reward"):
        latest_perf = latest_performance_by_agent(all_round_results)
        if selection_mode == "error_low":
            scores = {
                agent_id: float(latest_perf.get(agent_id, {}).get("avg_lateral_error", math.inf))
                for agent_id in candidate_agent_ids
            }
        else:
            scores = {
                agent_id: float(latest_perf.get(agent_id, {}).get("avg_reward", -math.inf))
                for agent_id in candidate_agent_ids
            }
    elif selection_mode == "cosine_high":
        ordered_deltas = [client_delta_by_agent_id[agent_id] for agent_id in candidate_agent_ids]
        direction = update_direction_metrics(ordered_deltas, old_global_weights)
        scores = {
            agent_id: float(direction["cosine_to_mean"][idx])
            for idx, agent_id in enumerate(candidate_agent_ids)
        }
    else:
        raise ValueError(f"Unknown client_selection_mode: {selection_mode}")

    if selection_mode == "count_balanced_reward":
        counts = selection_counts or {}
        ranked_agent_ids = sorted(
            candidate_agent_ids,
            key=lambda agent_id: (
                int(counts.get(agent_id, 0)),
                -scores[agent_id],
                agent_id,
            ),
        )
        return sorted(ranked_agent_ids[:budget]), scores

    if selection_mode == "error_low":
        ranked_agent_ids = sorted(candidate_agent_ids, key=lambda agent_id: (scores[agent_id], agent_id))
    else:
        ranked_agent_ids = sorted(candidate_agent_ids, key=lambda agent_id: (-scores[agent_id], agent_id))
    return sorted(ranked_agent_ids[:budget]), scores

def append_client_selection_logs(
    csv_dir,
    round_index,
    selection_mode,
    selection_budget,
    candidate_agent_ids,
    selected_agent_ids,
    selection_scores,
):
    if selection_mode in ("all", "single"):
        return

    selected_set = set(selected_agent_ids)
    agent_rows = []
    for agent_id in candidate_agent_ids:
        agent_rows.append({
            "round": round_index,
            "seed": seed,
            "condition": heterogeneity_condition or "standard",
            "algorithm": aggregation_method,
            "selection_mode": selection_mode,
            "selection_budget": selection_budget,
            "agent_id": agent_id,
            "selection_score": selection_scores.get(agent_id, 0.0),
            "selected_for_aggregation": int(agent_id in selected_set),
        })

    round_row = {
        "round": round_index,
        "seed": seed,
        "condition": heterogeneity_condition or "standard",
        "algorithm": aggregation_method,
        "selection_mode": selection_mode,
        "selection_budget": selection_budget,
        "candidate_agent_ids": ";".join(str(agent_id) for agent_id in candidate_agent_ids),
        "selected_agent_ids": ";".join(str(agent_id) for agent_id in selected_agent_ids),
    }

    agent_csv_path = os.path.join(csv_dir, "client_selection_agent_level.csv")
    round_csv_path = os.path.join(csv_dir, "client_selection_round_level.csv")
    pd.DataFrame(agent_rows, columns=CLIENT_SELECTION_AGENT_COLUMNS).to_csv(
        agent_csv_path,
        mode="a",
        header=not os.path.exists(agent_csv_path) or os.path.getsize(agent_csv_path) == 0,
        index=False,
    )
    pd.DataFrame([round_row], columns=CLIENT_SELECTION_ROUND_COLUMNS).to_csv(
        round_csv_path,
        mode="a",
        header=not os.path.exists(round_csv_path) or os.path.getsize(round_csv_path) == 0,
        index=False,
    )

# --------------------------------------------------------------------------------
# ✅ 1. 메인 프로세스
# --------------------------------------------------------------------------------
def main():
    set_global_seed(seed, context="Main")
    
    cx_list, cy_list, cyaw_list = load_direct_paths(mode=route_mode)

    # --- 결과 저장 경로 ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_subdir = getattr(config, "results_subdir", "fedadagrad")
    model_file_suffix = getattr(config, "model_file_suffix", "fedadagrad")
    if client_selection_mode == "single":
        selection_tag = f"single_agent{client_selection_agent_id}"
    else:
        selection_tag = f"{client_selection_mode}_{client_selection_budget}of{num_agents}"
    experiment_tag = f"{experiment_name}_{selection_tag}" if experiment_name else selection_tag
    experiment_dir = os.path.join(base_dir, "experiments", experiment_tag)
    csv_dir = os.path.join(experiment_dir, "metrics")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    state_dir = os.path.join(experiment_dir, "state")
    temp_dir = os.path.join(experiment_dir, "temp")
    model_dir = os.path.join(experiment_dir, "models")
    for output_dir in (csv_dir, checkpoint_dir, state_dir, temp_dir, model_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"[Main] Experiment directory: {experiment_dir}")
    csv_filename = f"fl_metrics_{model_file_suffix}_{num_agents}_{route_mode}_{experiment_tag}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    # --- 백업 파일 경로 ---
    backup_weight_path = os.path.join(state_dir, "global_model_latest.weights.h5")
    last_episode_file = os.path.join(state_dir, "last_episode.txt")
    server_m_path = os.path.join(state_dir, "server_m.pkl")
    server_v_path = os.path.join(state_dir, "server_v.pkl")
    window_models_path = os.path.join(state_dir, "window_models.pkl")
    selection_counts_path = os.path.join(state_dir, "selection_counts.json")
    
    # Configure the coordinator and every spawned worker consistently.
    configure_tensorflow_runtime("Main")
            
    global_l = deep_network(6, action_len)

    # FedAdagrad 파라미터
    server_eta = getattr(config, "server_eta", 0.1)
    server_beta1 = getattr(config, "server_beta1", 0.9)
    server_beta2 = getattr(config, "server_beta2", 0.99)
    server_tau = getattr(config, "server_tau", 1e-6)

    server_m = [np.zeros_like(w) for w in global_l.get_weights()]
    server_v = [np.zeros_like(w) for w in global_l.get_weights()]
    window_models = []

    start_episode = 0
    if os.path.exists(backup_weight_path) and os.path.exists(last_episode_file):
        print(f"[Main] : {backup_weight_path}")
        try:
            global_l.load_weights(backup_weight_path)
            with open(last_episode_file, 'r') as f:
                start_episode = int(f.read())
            print(f"[Main] {start_episode}epi start.")
            
            if os.path.exists(server_m_path) and os.path.exists(server_v_path):
                with open(server_m_path, 'rb') as f_m: server_m = pickle.load(f_m)
                with open(server_v_path, 'rb') as f_v: server_v = pickle.load(f_v)
            if window_average_enabled and os.path.exists(window_models_path):
                with open(window_models_path, 'rb') as f_w: window_models = pickle.load(f_w)
        except Exception as e:
            print(f"[Error] checkpoint load failed: {e}. Restart from episode 0.")
            start_episode = 0
    
    selection_counts = {agent_id: 0 for agent_id in range(num_agents)}
    if client_selection_mode.startswith("count_balanced_"):
        if start_episode == 0 and os.path.exists(selection_counts_path):
            try:
                os.remove(selection_counts_path)
            except Exception as e:
                print(f"[Main] stale selection count cleanup failed: {e}")
        elif start_episode > 0 and not os.path.exists(selection_counts_path):
            print("[Main] Selection count state is missing; count-balanced selection will resume from zero counts.")
        selection_counts = load_selection_counts(selection_counts_path)
        print(f"[Main] Count-balanced selection counts: {selection_counts}")

    if start_episode == 0 and not os.path.exists(csv_path):
        print("Create new CSV file.")
        ensure_metric_csv_header(csv_path)
    else:
        print("Append to existing CSV file.")
        ensure_metric_csv_header(csv_path)

    current_episode = start_episode
    
    while current_episode < n_episode:
        
        round_start_time = time.time()
        episodes_this_round = min(round_fre, n_episode - current_episode)
        if episodes_this_round <= 0: break
            
        print(f"\n============================================================================================")
        round_label = "Standalone training round" if client_selection_mode == "single" else "FL round"
        print(f"[Main] {round_label} started. (episode {current_episode} ~ {current_episode + episodes_this_round - 1})")
        if client_selection_mode == "single":
            participating_agent_ids = [client_selection_agent_id]
            print(f"[Main] Standalone single-vehicle training. (agent={client_selection_agent_id})")
        else:
            participating_agent_ids = list(range(num_agents))
            print(f"[Main] Client selection: mode={client_selection_mode}, budget={client_selection_budget}")
        print(f"[Main] Participating agents: {participating_agent_ids}")
        if window_average_enabled:
            print(f"[Main] Window-Based Model Averaging enabled. (window={window_average_size}, stored={len(window_models)})")
        
        global_l.save_weights(backup_weight_path) 
        old_global_weights = global_l.get_weights()
        
        processes = []
        for agent_id in participating_agent_ids:
            p = multiprocessing.Process(
                target=worker_process, 
                args=(
                    agent_id, 
                    current_episode, 
                    episodes_this_round, 
                    backup_weight_path,
                    cx_list, 
                    cy_list, 
                    cyaw_list,
                    temp_dir
                )
            )
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        print(f"[Main] workers finished. (elapsed: {time.time() - round_start_time:.2f}s)")

        client_delta_by_agent_id = {}
        all_round_results = []
        
        temp_local_models = {}
        successful_agent_ids = []
        
        for agent_id in participating_agent_ids:
            local_weight_path = os.path.join(temp_dir, f"local_weights_{agent_id}.weights.h5")
            try:
                temp_local_models[agent_id] = deep_network(6, action_len)
                temp_local_models[agent_id].load_weights(local_weight_path)
                local_weights = temp_local_models[agent_id].get_weights()
                delta_i = [local_w - old_global_w for local_w, old_global_w in zip(local_weights, old_global_weights)]
                client_delta_by_agent_id[agent_id] = delta_i
                successful_agent_ids.append(agent_id)
            except Exception as e:
                print(f"[Error] Worker {agent_id} weight load failed: {e}")

            temp_csv_path = os.path.join(temp_dir, f"temp_results_{agent_id}.csv")
            try:
                if os.path.exists(temp_csv_path):
                    df = pd.read_csv(temp_csv_path)
                    all_round_results.append(df)
                    os.remove(temp_csv_path)
            except Exception as e:
                print(f"[Error] Worker {agent_id} CSV load failed: {e}")

        # Federated aggregation update
        if client_delta_by_agent_id:
            round_index = current_episode // max(round_fre, 1)
            reset_buffer_after_update = False

            if client_selection_mode == "single":
                selected_agent_id = successful_agent_ids[0]
                global_l.set_weights(temp_local_models[selected_agent_id].get_weights())
                print(f"[Main] Standalone single-vehicle update completed. (agent={selected_agent_id}, no federated aggregation)")
            else:
                selected_agent_ids, selection_scores = select_aggregation_clients(
                    client_selection_mode,
                    client_selection_budget,
                    successful_agent_ids,
                    client_delta_by_agent_id,
                    old_global_weights,
                    all_round_results,
                    round_index,
                    selection_counts,
                )
                client_deltas = [client_delta_by_agent_id[agent_id] for agent_id in selected_agent_ids]
                if not client_deltas:
                    print("[Main] No clients selected for aggregation.")
                else:
                    avg_delta_t = average_client_deltas(client_deltas, old_global_weights)
                    print(f"[Main] Aggregation-selected agents: {selected_agent_ids}")
                    append_client_selection_logs(
                        csv_dir,
                        round_index,
                        client_selection_mode,
                        client_selection_budget,
                        successful_agent_ids,
                        selected_agent_ids,
                        selection_scores,
                    )
                    append_heterogeneity_logs(
                        csv_dir,
                        round_index,
                        selected_agent_ids,
                        client_deltas,
                        old_global_weights,
                        all_round_results,
                        temp_dir,
                    )
                    if client_selection_mode.startswith("count_balanced_"):
                        selection_counts = update_selection_counts(selection_counts, selected_agent_ids)
                        save_selection_counts(selection_counts_path, selection_counts)
                        print(f"[Main] Count-balanced selection counts: {selection_counts}")

                    if aggregation_method == "cosine_fedavg":
                        weighted_delta_t, cosine_weights, _ = cosine_aware_delta(
                            client_deltas,
                            old_global_weights,
                            lambda_value=cosine_lambda,
                        )
                        new_global_weights = [
                            old_w + delta for old_w, delta in zip(old_global_weights, weighted_delta_t)
                        ]
                        print(f"[Main] Cosine-aware FedAvg weights: {[round(w, 4) for w in cosine_weights]}")
                    else:
                        new_global_weights, server_m, server_v = apply_aggregation(
                            old_global_weights,
                            avg_delta_t,
                            server_m,
                            server_v,
                        )

                    if window_average_enabled:
                        window_models = append_window_model(window_models, new_global_weights, window_average_size)
                        new_global_weights = average_window_models(window_models)
                        print(f"[Main] Window-Based Model Averaging applied. (models={len(window_models)}/{window_average_size})")

                    global_l.set_weights(new_global_weights)
                    if aggregation_method == "fedavg":
                        print("[Main] FedAvg update completed.")
                    elif aggregation_method == "fedprox":
                        print(f"[Main] FedProx update completed. (mu={fedprox_mu})")
                    elif aggregation_method == "cosine_fedavg":
                        print(f"[Main] Cosine-aware FedAvg update completed. (lambda={cosine_lambda})")
                    elif aggregation_method == "fedadam":
                        print("[Main] FedAdam update completed.")
                    else:
                        print("[Main] FedAdagrad update completed.")

                    reset_buffer_after_update = True

            if reset_buffer_after_update:
                reset_replay_buffers(temp_dir, participating_agent_ids)
                print("[Main] Replay buffers reset after global model update.")

            for agent_id in participating_agent_ids:
                for filename in (f"local_weights_{agent_id}.weights.h5", f"local_target_weights_{agent_id}.weights.h5"):
                    if client_selection_mode == "single" and filename.startswith("local_target_weights_"):
                        continue
                    temp_weight_path = os.path.join(temp_dir, filename)
                    if os.path.exists(temp_weight_path):
                        try:
                            os.remove(temp_weight_path)
                        except Exception as e:
                            print(f"[Main] temp weight cleanup failed: {temp_weight_path}: {e}")

            del temp_local_models, client_delta_by_agent_id, old_global_weights
            K.clear_session()
            gc.collect()

        # CSV 저장
        if all_round_results:
            new_rows_df = pd.concat(all_round_results, ignore_index=True)
            new_rows_df = new_rows_df.sort_values(by=['episode', 'path_id'])
            
            header_needed = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            new_rows_df = new_rows_df[METRIC_COLUMNS]
            new_rows_df.to_csv(csv_path, mode='a', header=header_needed, index=False)
            
            print('============================================================================================')
            last_episode_data = new_rows_df[new_rows_df['episode'] == (current_episode + episodes_this_round - 1)]
            for _, row in last_episode_data.iterrows():
                print(f"{int(row['episode'])} EP / Agent {int(row['path_id'])} : Avg Lat Err {row['avg_lateral_error']:.4f}")
            print('============================================================================================')

        global_l.save_weights(backup_weight_path) 
        current_episode += episodes_this_round
        with open(last_episode_file, 'w') as f:
            f.write(str(current_episode))

        completed_round = (current_episode + max(round_fre, 1) - 1) // max(round_fre, 1)
        if completed_round % checkpoint_every_rounds == 0 or current_episode >= n_episode:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"global_model_episode_{current_episode:06d}.weights.h5",
            )
            global_l.save_weights(checkpoint_path)
            print(f"[Main] Model checkpoint saved: {checkpoint_path}")
            
        try:
            with open(server_m_path, 'wb') as f_m: pickle.dump(server_m, f_m)
            with open(server_v_path, 'wb') as f_v: pickle.dump(server_v, f_v)
            if window_average_enabled:
                with open(window_models_path, 'wb') as f_w: pickle.dump(window_models, f_w)
        except: pass

        gc.collect()

    print("Total training finished.")
    final_model_prefix = "single_vehicle_model_final" if client_selection_mode == "single" else "fl_global_model_final"
    final_model_path = os.path.join(model_dir, f"{final_model_prefix}_{model_file_suffix}.h5")
    global_l.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()


