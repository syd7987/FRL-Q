import os
import numpy as np
from collections import deque

show_animation = True
save_image = True


rho = 0.1
lamda = 0.99
eps = 0.1
initial_eps = eps
eps_decay = 0.99
batch_siz = 64
D_h = deque(maxlen=3000)
train_episodes = 1000
test_episodes = 10000
n_episode = train_episodes
timestep = 200
seed = int(os.environ.get("FL_SEED", 41))
t_up = 100

# --- Runtime performance options ---
# DQN workers may use the GPU. Set FL_WORKER_DEVICE=cpu for CPU-only runs.
worker_device = os.environ.get("FL_WORKER_DEVICE", "gpu").strip().lower()
if worker_device not in ("cpu", "gpu"):
    raise ValueError(f"Unknown FL_WORKER_DEVICE: {worker_device}")
cpu_threads_per_worker = max(1, int(os.environ.get("FL_CPU_THREADS_PER_WORKER", "1")))
mixed_precision_enabled = os.environ.get("FL_MIXED_PRECISION", "1").lower() in (
    "1", "true", "yes", "on"
)
# --- Federated Route Experiment Options ---
# (num_agents / route_mode / agent_route_assignments 는 experiment_profile에서 로드됩니다.)

# 경로 좌표 정의를 config에서 관리
train_paths = [
    {'AX': [0.0, -14.2539, -20.1706, -4.0341, -38.4585, -20.4395, -48.4093], 'AY': [0.0, 0.4267, -1.7066, -30.7194, -40.7461, -71.8931, -79.5733]},
    {'AX': [0.0, -8.5175, -50.0829, -93.6926], 'AY': [0.0, -27.0272, -39.7302, 40.2698]},
    {'AX': [0.0, 9.1428, -13.1429, 2.2857, 86.8571], 'AY': [0.0, -16.3183, -22.891, -51.6752, -23.7975]},
    {'AX': [0.0, -3.1189, -17.934, 5.7181, -1.8194, 18.9737, 25.9914, 45.0551, 32.2293], 'AY': [0.0, 13.4024, 39.1759, 45.7738, 59.7943, 64.9488, 51.3408, 57.9387, 80.0]},
    {'AX': [0.0, 63.1897, 29.0673, -91.6251], 'AY': [0.0, 17.5444, 106.26741, 71.6808]},
    {'AX': [0.0, 128.0177, 171.4324], 'AY': [0.0, 44.5932, -35.3218]},
    {'AX': [0.0, 57.3291, 41.7449, 86.2728, 20.0375], 'AY': [0.0, 16.3374, 42.3888, 55.1936, 170.8762]},
    {'AX': [0.0, -16.1413, 122.4517], 'AY': [0.0, 31.7906, 72.8531]},
]

test_path = {'AX': [0.0, -9.52, 18.15, 15.5, 53.58, 62.88, 67.09], 'AY': [0.0, 27.35, 29.34, 58.05, 62.24, 47.46, 32.68]}

# 실험 프리셋: 실험명만 바꿔서 세팅 전환
experiment_profile = os.environ.get("FL_EXPERIMENT_PROFILE", "non_iid_4_train_fedadagrad")
EXPERIMENT_PRESETS = {
    "non_iid_8_train_fedavg": {
        "num_agents": 8,
        "route_mode": "train",
        "agent_route_assignments": [0, 1, 2, 3, 4, 5, 6, 7],
        "aggregation_method": "fedavg",
        "results_subdir": "fedavg/non_iid_8_train",
        "model_file_suffix": "fedavg",
    },
    "non_iid_8_test_fedavg": {
        "num_agents": 8,
        "route_mode": "test",
        "agent_route_assignments": [0] * 8,
        "aggregation_method": "fedavg",
        "results_subdir": "fedavg/non_iid_8_test",
        "model_file_suffix": "fedavg",
    },
    "non_iid_8_train_fedadam": {
        "num_agents": 8,
        "route_mode": "train",
        "agent_route_assignments": [0, 1, 2, 3, 4, 5, 6, 7],
        "aggregation_method": "fedadam",
        "results_subdir": "fedadam/non_iid_8_train",
        "model_file_suffix": "fedadam",
    },
    "non_iid_8_test_fedadam": {
        "num_agents": 8,
        "route_mode": "test",
        "agent_route_assignments": [0] * 8,
        "aggregation_method": "fedadam",
        "results_subdir": "fedadam/non_iid_8_test",
        "model_file_suffix": "fedadam",
    },
    "non_iid_8_train_fedadagrad": {
        "num_agents": 8,
        "route_mode": "train",
        "agent_route_assignments": [0, 1, 2, 3, 4, 5, 6, 7],
        "aggregation_method": "fedadagrad",
        "results_subdir": "fedadagrad/non_iid_8_train",
        "model_file_suffix": "fedadagrad",
    },
    "non_iid_8_test_fedadagrad": {
        "num_agents": 8,
        "route_mode": "test",
        "agent_route_assignments": [0] * 8,
        "aggregation_method": "fedadagrad",
        "results_subdir": "fedadagrad/non_iid_8_test",
        "model_file_suffix": "fedadagrad",
    },
}

for _method in ["fedavg", "fedadam", "fedadagrad", "fedprox", "cosine_fedavg"]:
    for _num_agents in [1, 2, 4, 8]:
        for _mode, _episodes in [("train", train_episodes), ("test", test_episodes)]:
            EXPERIMENT_PRESETS[f"non_iid_{_num_agents}_{_mode}_{_method}"] = {
                "num_agents": _num_agents,
                "route_mode": _mode,
                "agent_route_assignments": list(range(_num_agents)) if _mode == "train" else [0] * _num_agents,
                "aggregation_method": _method,
                "results_subdir": f"{_method}/non_iid_{_num_agents}_{_mode}",
                "model_file_suffix": _method,
                "n_episode": _episodes,
            }
            EXPERIMENT_PRESETS[f"non_iid_{_num_agents}_{_mode}_{_method}_wma"] = {
                "num_agents": _num_agents,
                "route_mode": _mode,
                "agent_route_assignments": list(range(_num_agents)) if _mode == "train" else [0] * _num_agents,
                "aggregation_method": _method,
                "window_average_enabled": True,
                "window_average_size": 5,
                "results_subdir": f"{_method}_wma/non_iid_{_num_agents}_{_mode}",
                "model_file_suffix": f"{_method}_wma",
                "n_episode": _episodes,
            }

def apply_experiment_profile(profile_name):
    if profile_name not in EXPERIMENT_PRESETS:
        raise ValueError(f"Unknown experiment_profile: {profile_name}")

    preset = EXPERIMENT_PRESETS[profile_name]
    globals()["num_agents"] = preset["num_agents"]
    globals()["route_mode"] = preset["route_mode"]
    globals()["agent_route_assignments"] = list(preset["agent_route_assignments"])

    optional_keys = [
        "aggregation_method",
        "results_subdir",
        "model_file_suffix",
        "server_eta",
        "server_beta1",
        "server_beta2",
        "server_tau",
        "window_average_enabled",
        "window_average_size",
        "n_episode",
    ]
    for key in optional_keys:
        if key in preset:
            globals()[key] = preset[key]


apply_experiment_profile(experiment_profile)

# --- Runtime / Logging Options ---
round_fre = 10
checkpoint_every_rounds = max(1, int(os.environ.get("FL_CHECKPOINT_EVERY_ROUNDS", "1")))
results_subdir = globals().get("results_subdir", "fedadagrad")
model_file_suffix = globals().get("model_file_suffix", "fedadagrad")
aggregation_method = globals().get("aggregation_method", "fedadagrad")  # "fedavg" | "fedprox" | "fedadagrad" | "fedadam"
window_average_enabled = globals().get("window_average_enabled", False)
window_average_size = int(globals().get("window_average_size", 5))

# --- FedProx ---
fedprox_mu = float(os.environ.get("FL_FEDPROX_MU", globals().get("fedprox_mu", 0.01)))

# --- Server Optimizer (FedAdagrad/FedAdam) ---
server_eta = globals().get("server_eta", 0.1)
server_beta1 = globals().get("server_beta1", 0.9)
server_beta2 = globals().get("server_beta2", 0.99)
server_tau = globals().get("server_tau", 1e-6)


dt = 0.1
target_vel = 22.22 # m/s

n_episode = int(os.environ.get("FL_N_EPISODE", n_episode))
experiment_name = os.environ.get("FL_EXPERIMENT_NAME", f"ep{n_episode}")
window_average_enabled = os.environ.get("FL_WINDOW_AVERAGE", str(window_average_enabled)).lower() in ("1", "true", "yes", "on")
window_average_size = int(os.environ.get("FL_WINDOW_SIZE", window_average_size))
heterogeneity_condition = os.environ.get("FL_HETEROGENEITY_CONDITION", "").strip().lower()
heterogeneity_log_enabled = os.environ.get("FL_HETEROGENEITY_LOG", "0").lower() in ("1", "true", "yes", "on")
cosine_lambda = float(os.environ.get("FL_COSINE_LAMBDA", "1.0"))
client_selection_mode = (
    os.environ.get("FL_CLIENT_SELECTION_MODE", "all")
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
allowed_client_selection_modes = (
    "all",
    "single",
    "random",
    "error_low",
    "reward_high",
    "cosine_high",
    "count_balanced_random",
    "count_balanced_reward",
)
if client_selection_mode not in allowed_client_selection_modes:
    raise ValueError(f"Unknown FL_CLIENT_SELECTION_MODE: {client_selection_mode}")
client_selection_agent_id = int(os.environ.get("FL_CLIENT_SELECTION_AGENT_ID", "0"))
client_selection_agent_id = max(0, min(client_selection_agent_id, num_agents - 1))
default_client_selection_budget = num_agents if client_selection_mode == "all" else min(2, num_agents)
if client_selection_mode == "single":
    default_client_selection_budget = 1
client_selection_budget = int(os.environ.get("FL_CLIENT_SELECTION_BUDGET", default_client_selection_budget))
client_selection_budget = max(1, min(client_selection_budget, num_agents))
if client_selection_mode == "single":
    client_selection_budget = 1

if heterogeneity_condition:
    allowed_heterogeneity_conditions = (
        "homogeneous",
        "same_vehicle_diff_path",
        "diff_vehicle_same_path",
        "combined",
    )
    if heterogeneity_condition not in allowed_heterogeneity_conditions:
        raise ValueError(f"Unknown FL_HETEROGENEITY_CONDITION: {heterogeneity_condition}")
    route_mode = "train"
    if heterogeneity_condition == "homogeneous":
        agent_route_assignments = [0] * num_agents
        vehicle_mode = "homogeneous"
    elif heterogeneity_condition == "same_vehicle_diff_path":
        agent_route_assignments = list(range(num_agents))
        vehicle_mode = "homogeneous"
    elif heterogeneity_condition == "diff_vehicle_same_path":
        agent_route_assignments = [0] * num_agents
        vehicle_mode = "combined"
    else:
        agent_route_assignments = list(range(num_agents))
        vehicle_mode = "combined"
    results_subdir = f"heterogeneity/{heterogeneity_condition}/{aggregation_method}/K{num_agents}"
    model_file_suffix = aggregation_method
else:
    vehicle_mode = os.environ.get("FL_VEHICLE_MODE", "combined").strip().lower()

results_subdir = os.environ.get("FL_RESULTS_SUBDIR", results_subdir)
model_file_suffix = os.environ.get("FL_MODEL_FILE_SUFFIX", model_file_suffix)

# --- Action Space ---
Acceleration = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
Delta = [np.radians(delt - 40) for delt in range(81)]


class Vehicle:
    def __init__(self, L, Lf, Lr, m, Cf, Cr, Iz, max_steer, noise_Var, noise_Var_xy):
        self.L = L
        self.Lf = Lf
        self.Lr = Lr
        self.m = m
        self.Cf = Cf
        self.Cr = Cr
        self.Iz = Iz
        self.max_steer = max_steer
        self.noise_Var = noise_Var
        self.noise_Var_xy = noise_Var_xy


vehicle_parameters = [
    {
     # Audi A3
        'L': 2.6, 'Lf': 1.55, 'Lr': 1.05, 'm': 1400, 'Cf': 85000, 'Cr': 90000,
        'Iz': 2200, 'max_steer': np.radians(40.0), 'noise_Var': 1e-3, 'noise_Var_xy': 1e-3
    },
    {
     # Toyota Prius
        'L': 2.7, 'Lf': 1.65, 'Lr': 1.05, 'm': 1450, 'Cf': 70000, 'Cr': 70000,
        'Iz': 2750, 'max_steer': np.radians(40.0), 'noise_Var': 1e-3, 'noise_Var_xy': 1e-3

    },
    {
     # Porsche 911
         'L': 2.45, 'Lf': 1.42, 'Lr': 0.99, 'm': 1640, 'Cf': 90000, 'Cr': 90000,
         'Iz': 2361, 'max_steer': np.radians(40.0), 'noise_Var': 1e-3, 'noise_Var_xy': 1e-3
    },
    {
     # Hyundai Azera
        'L': 2.7, 'Lf': 1.65, 'Lr': 1.05, 'm': 1450, 'Cf': 70000, 'Cr': 70000,
        'Iz': 2750, 'max_steer': np.radians(40.0), 'noise_Var': 1e-3, 'noise_Var_xy': 1e-3

    }
]


if num_agents > len(vehicle_parameters):
    print(f"Warning: num_agents ({num_agents}) is greater than the number of available vehicle parameters ({len(vehicle_parameters)}). Cycling vehicle templates.")

if vehicle_mode == "homogeneous":
    vehicles = [Vehicle(**vehicle_parameters[0]) for _ in range(num_agents)]
else:
    vehicles = [Vehicle(**vehicle_parameters[i % len(vehicle_parameters)]) for i in range(num_agents)]
