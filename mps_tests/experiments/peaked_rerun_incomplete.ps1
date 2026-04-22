# Rerun peaked overlap benchmarks still "open" under relaxed completion:
# OK if (successful bond_dim=1024) OR (any bond with dominant_overlap_percent>=100).
# This script only lists jobs that satisfy neither (plus never-started circuits 688/1271).
# Prereq: repo root as cwd, BLUEQUBIT_API_TOKEN set, Python env with bluequbit + qiskit.
#
# Usage (from repo root):
#   pwsh ./mps_tests/experiments/peaked_rerun_incomplete.ps1 -List
#   pwsh ./mps_tests/experiments/peaked_rerun_incomplete.ps1 -All
#   pwsh ./mps_tests/experiments/peaked_rerun_incomplete.ps1 -Id CPU_619

param(
    [switch] $List,
    [switch] $All,
    [string] $Id = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Runner = Join-Path $RepoRoot "mps_tests\experiments\peaked_overlap_bond_dimension.py"

# Each entry: stable id, device, RZZ count (unique per input-peaked-circuits file).
$Jobs = @(
    @{ Id = "CPU_619";  Device = "mps.cpu"; Rzz = 619 }
    @{ Id = "CPU_1245"; Device = "mps.cpu"; Rzz = 1245 }
    @{ Id = "CPU_835";  Device = "mps.cpu"; Rzz = 835 }
    @{ Id = "CPU_1320"; Device = "mps.cpu"; Rzz = 1320 }
    @{ Id = "GPU_1320"; Device = "mps.gpu"; Rzz = 1320 }
    @{ Id = "CPU_927";  Device = "mps.cpu"; Rzz = 927 }
    @{ Id = "CPU_1438"; Device = "mps.cpu"; Rzz = 1438 }
    @{ Id = "GPU_1438"; Device = "mps.gpu"; Rzz = 1438 }
    @{ Id = "CPU_688";  Device = "mps.cpu"; Rzz = 688 }
    @{ Id = "GPU_688";  Device = "mps.gpu"; Rzz = 688 }
    @{ Id = "CPU_1271"; Device = "mps.cpu"; Rzz = 1271 }
    @{ Id = "GPU_1271"; Device = "mps.gpu"; Rzz = 1271 }
)

function Invoke-PeakedJob {
    param(
        [string] $Device,
        [int] $Rzz
    )
    python $Runner --device $Device --rzz-gates $Rzz
}

if ($List) {
    $Jobs | ForEach-Object { Write-Output "$($_.Id)  ->  python mps_tests/experiments/peaked_overlap_bond_dimension.py --device $($_.Device) --rzz-gates $($_.Rzz)" }
    exit 0
}

if ($All) {
    foreach ($j in $Jobs) {
        Write-Host "`n========== $($j.Id) ==========" -ForegroundColor Cyan
        Invoke-PeakedJob -Device $j.Device -Rzz $j.Rzz
    }
    exit $LASTEXITCODE
}

if ($Id) {
    $j = $Jobs | Where-Object { $_.Id -eq $Id }
    if (-not $j) {
        Write-Error "Unknown Id '$Id'. Use -List to see ids."
        exit 2
    }
    Invoke-PeakedJob -Device $j.Device -Rzz $j.Rzz
    exit $LASTEXITCODE
}

Write-Host "Use -List, -All, or -Id <id> (see -List)." -ForegroundColor Yellow
exit 1
