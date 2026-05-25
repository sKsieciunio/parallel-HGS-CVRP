# benchmark.ps1
# Quick evaluation: runs HGS-CVRP with and without GPU acceleration.
#
# Usage:
#   .\benchmark.ps1 -ExePath <path\to\hgs.exe> [options]
#
# Options:
#   -ExePath      Path to the hgs executable (required)
#   -Instance     Path to .vrp instance (default: Instances\CVRP\X-n1001-k43.vrp)
#   -Seed         Fixed seed for reproducibility (default: 1)
#   -TimeLimit    Time limit in seconds per run (default: 10)
#   -Runs         Number of repeated runs per mode (default: 3)

param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [string]$Instance = "$PSScriptRoot\..\Instances\CVRP\X-n1001-k43.vrp",
    [int]$Seed = 1,
    [double]$TimeLimit = 10,
    [int]$Runs = 3
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) {
    Write-Error "Executable not found: $ExePath"
    exit 1
}
if (-not (Test-Path $Instance)) {
    Write-Error "Instance file not found: $Instance"
    exit 1
}

$InstanceName = [System.IO.Path]::GetFileNameWithoutExtension($Instance)
$TmpSol = "$env:TEMP\hgs_bench_tmp.sol"

function Run-Mode {
    param([string]$Label, [int]$GpuFlag)

    $results = @()

    for ($i = 1; $i -le $Runs; $i++) {
        $runSeed = $Seed + $i - 1
        $args = @($Instance, $TmpSol, "-seed", $runSeed, "-t", $TimeLimit, "-gpu", $GpuFlag, "-log", 1)

        $output = & $ExePath @args 2>&1

        # Parse iterations and CPU time from the FINISHED line.
        $nbIter   = $null
        $timeSpent = $null
        $finishedLine = $output | Where-Object { $_ -match "GENETIC ALGORITHM FINISHED" } | Select-Object -Last 1
        if ($finishedLine -match "FINISHED AFTER\s+(\d+)\s+ITERATIONS") { $nbIter    = [int]$matches[1] }
        if ($finishedLine -match "TIME SPENT:\s*([\d.]+)")              { $timeSpent = [double]$matches[1] }

        # Parse best feasible cost from the PG.csv progress file.
        # Format: instanceName;seed;cost;timeSeconds — cost is the 3rd field.
        $bestCost = $null
        $pgFile = "$TmpSol.PG.csv"
        if (Test-Path $pgFile) {
            $lastLine = Get-Content $pgFile | Select-Object -Last 1
            if ($lastLine -match "^[^;]+;\d+;([\d.]+)") {
                $bestCost = [double]$matches[1]
            }
            Remove-Item $pgFile -ErrorAction SilentlyContinue
        }

        $results += [PSCustomObject]@{
            Run      = $i
            Seed     = $runSeed
            BestCost = $bestCost
            Iters    = $nbIter
            TimeSec  = $timeSpent
        }

        $costStr = if ($null -ne $bestCost) { "{0:F2}" -f $bestCost } else { "N/A" }
        $iterStr = if ($null -ne $nbIter)   { "$nbIter" }             else { "N/A" }
        $timeStr = if ($null -ne $timeSpent) { "{0:F2}s" -f $timeSpent } else { "N/A" }
        Write-Host ("  [{0}] run {1}/{2}  seed={3}  best={4}  iters={5}  time={6}" -f $Label, $i, $Runs, $runSeed, $costStr, $iterStr, $timeStr)
    }

    return $results
}

function Summarize {
    param([object[]]$Results, [string]$Label)

    $costs = $Results | Where-Object { $null -ne $_.BestCost } | ForEach-Object { $_.BestCost }
    $times = $Results | Where-Object { $null -ne $_.TimeSec  } | ForEach-Object { $_.TimeSec  }
    $iters = $Results | Where-Object { $null -ne $_.Iters    } | ForEach-Object { $_.Iters    }

    $avgCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Average).Average } else { $null }
    $minCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Minimum).Minimum } else { $null }
    $avgTime = if ($times.Count -gt 0) { ($times | Measure-Object -Average).Average } else { $null }
    $avgIter = if ($iters.Count -gt 0) { ($iters | Measure-Object -Average).Average } else { $null }

    return [PSCustomObject]@{
        Mode    = $Label
        Runs    = $Results.Count
        AvgCost = $avgCost
        MinCost = $minCost
        AvgTime = $avgTime
        AvgIter = $avgIter
    }
}

# ── Header ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  HGS-CVRP Quick Benchmark  |  instance: {0}" -f $InstanceName) -ForegroundColor Cyan
Write-Host ("  {0} runs x {1}s  |  seed base: {2}" -f $Runs, $TimeLimit, $Seed) -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── CPU runs ──────────────────────────────────────────────────────────────────
Write-Host "--- CPU (gpu=0) ---" -ForegroundColor Yellow
$cpuResults = Run-Mode -Label "CPU" -GpuFlag 0

Write-Host ""

# ── GPU runs ──────────────────────────────────────────────────────────────────
Write-Host "--- GPU (gpu=1) ---" -ForegroundColor Green
$gpuResults = Run-Mode -Label "GPU" -GpuFlag 1

Write-Host ""

# ── Summary ───────────────────────────────────────────────────────────────────
$cpuSummary = Summarize -Results $cpuResults -Label "CPU"
$gpuSummary = Summarize -Results $gpuResults -Label "GPU"

Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$fmt = "{0,-6} | {1,10} | {2,10} | {3,10} | {4,10}"
Write-Host ($fmt -f "Mode", "Avg Cost", "Min Cost", "Avg Time(s)", "Avg Iters")
Write-Host ("-" * 58)

foreach ($s in @($cpuSummary, $gpuSummary)) {
    $avgCostStr = if ($null -ne $s.AvgCost) { "{0:F2}" -f $s.AvgCost }           else { "N/A" }
    $minCostStr = if ($null -ne $s.MinCost) { "{0:F2}" -f $s.MinCost }           else { "N/A" }
    $avgTimeStr = if ($null -ne $s.AvgTime) { "{0:F2}" -f $s.AvgTime }           else { "N/A" }
    $avgIterStr = if ($null -ne $s.AvgIter) { "{0:F0}" -f $s.AvgIter }           else { "N/A" }
    Write-Host ($fmt -f $s.Mode, $avgCostStr, $minCostStr, $avgTimeStr, $avgIterStr)
}

# ── Speedup ───────────────────────────────────────────────────────────────────
if ($null -ne $cpuSummary.AvgTime -and $null -ne $gpuSummary.AvgTime -and $gpuSummary.AvgTime -gt 0) {
    $speedup = $cpuSummary.AvgTime / $gpuSummary.AvgTime
    Write-Host ""
    Write-Host ("  Speedup (CPU/GPU avg time): {0:F2}x" -f $speedup) -ForegroundColor Magenta
}

Write-Host ""

# Cleanup
Remove-Item $TmpSol -ErrorAction SilentlyContinue
