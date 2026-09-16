$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace '.venv-gpu\Scripts\python.exe'

$env:FL_SEED = '41'
$env:FL_N_EPISODE = '1000'
$env:FL_CHECKPOINT_EVERY_ROUNDS = '1'
$env:FL_WORKER_DEVICE = 'gpu'
$env:FL_MIXED_PRECISION = '1'
$env:FL_CPU_THREADS_PER_WORKER = '1'
$env:MPLBACKEND = 'Agg'

$runs = @(
    @{
        Profile = 'non_iid_4_train_fedadagrad'
        Mode = 'all'
        Budget = '4'
        Name = 'optimized_fedadagrad_4all_20260916_run1'
        Tag = 'all_4of4'
    },
    @{
        Profile = 'non_iid_8_train_fedadagrad'
        Mode = 'error_low'
        Budget = '4'
        Name = 'optimized_fedadagrad_8errorlow4_20260916_run1'
        Tag = 'error_low_4of8'
    },
    @{
        Profile = 'non_iid_8_train_fedadagrad'
        Mode = 'random'
        Budget = '4'
        Name = 'optimized_fedadagrad_8random4_20260916_run1'
        Tag = 'random_4of8'
    }
)

foreach ($run in $runs) {
    $env:FL_EXPERIMENT_PROFILE = $run.Profile
    $env:FL_CLIENT_SELECTION_MODE = $run.Mode
    $env:FL_CLIENT_SELECTION_BUDGET = $run.Budget
    $env:FL_EXPERIMENT_NAME = $run.Name

    $experimentDir = Join-Path $workspace "experiments\$($run.Name)_$($run.Tag)"
    $logDir = Join-Path $experimentDir 'logs'
    $snapshotDir = Join-Path $experimentDir 'source_snapshot'
    New-Item -ItemType Directory -Force -Path $logDir, $snapshotDir | Out-Null
    Copy-Item -LiteralPath (
        Join-Path $workspace 'FLJDQN.py'
    ), (
        Join-Path $workspace 'FLJDQN_config.py'
    ), (
        Join-Path $workspace 'cubic_spline_planner.py'
    ) -Destination $snapshotDir -Force

    $stdout = Join-Path $logDir 'stdout.log'
    $stderr = Join-Path $logDir 'stderr.log'
    & $python -u (Join-Path $workspace 'FLJDQN.py') 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed with exit code ${LASTEXITCODE}: $($run.Name)"
    }
}
