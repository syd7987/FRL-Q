$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $workspace '.venv-gpu\Scripts\python.exe'
$entrypoint = Join-Path $workspace 'FLJDQN.py'
$launcherDir = Join-Path $workspace 'experiments\_fast_parallel_launcher'
New-Item -ItemType Directory -Force -Path $launcherDir | Out-Null

# Conservative process-level limits prevent TensorFlow's per-client thread pools
# from exhausting Windows resources when all three experiments run together.
$baseEnvironment = @{
    FL_SEED = '41'
    FL_N_EPISODE = '1000'
    FL_CHECKPOINT_EVERY_ROUNDS = '5'
    FL_WORKER_DEVICE = 'gpu'
    FL_MIXED_PRECISION = '1'
    FL_CPU_THREADS_PER_WORKER = '1'
    OMP_NUM_THREADS = '1'
    TF_NUM_INTRAOP_THREADS = '1'
    TF_NUM_INTEROP_THREADS = '1'
    TF_FORCE_GPU_ALLOW_GROWTH = 'true'
    MPLBACKEND = 'Agg'
}

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
        Name = 'fast_parallel_fedadagrad_8errorlow4_20260916_run1'
        Tag = 'error_low_4of8'
    },
    @{
        Profile = 'non_iid_8_train_fedadagrad'
        Mode = 'random'
        Budget = '4'
        Name = 'fast_parallel_fedadagrad_8random4_20260916_run1'
        Tag = 'random_4of8'
    }
)

$processes = @()
foreach ($run in $runs) {
    foreach ($item in $baseEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($item.Key, $item.Value, 'Process')
    }
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
    $process = Start-Process -FilePath $python `
        -ArgumentList @('-u', $entrypoint) `
        -WorkingDirectory $workspace `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    $processes += [pscustomobject]@{
        Name = $run.Name
        Pid = $process.Id
        Process = $process
    }
    Start-Sleep -Seconds 8
}

$processes | Select-Object Name, Pid | Export-Csv `
    (Join-Path $launcherDir 'coordinator_pids.csv') -NoTypeInformation

$failed = @()
foreach ($item in $processes) {
    $item.Process.WaitForExit()
    if ($item.Process.ExitCode -ne 0) {
        $failed += "$($item.Name)=$($item.Process.ExitCode)"
    }
}

if ($failed.Count -gt 0) {
    throw "Parallel experiment failure(s): $($failed -join ', ')"
}

