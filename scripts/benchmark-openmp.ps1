# benchmark-openmp.ps1
# Compares CPU baseline (sequential SWAP*) against OpenMP parallel SWAP* at
# increasing thread counts.  Thread count is controlled via OMP_NUM_THREADS;
# the -openMP flag enables the parallel code path.
#
# Usage:
#   .\benchmark-openmp.ps1 -ExePath <path\to\hgs.exe> [options]
#
# Options:
#   -ExePath      Path to the hgs executable (required)
#   -Instance     Path to .vrp instance (default: X-n1001-k43.vrp)
#   -Seed         Fixed seed for reproducibility (default: 1)
#   -TimeLimit    Time limit in seconds per run (default: 10)
#   -Runs         Number of repeated runs per configuration (default: 3)
#   -Threads      Comma-separated list of thread counts to test (default: 1,2,4,8)

param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [string]$Instance = "$PSScriptRoot\..\Instances\CVRP\X-n1001-k43.vrp",
    [int]$Seed = 1,
    [double]$TimeLimit = 10,
    [int]$Runs = 3,
    [int[]]$Threads = @(1, 2, 4, 8)
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) { Write-Error "Executable not found: $ExePath"; exit 1 }
if (-not (Test-Path $Instance)) { Write-Error "Instance file not found: $Instance"; exit 1 }

$InstanceName = [System.IO.Path]::GetFileNameWithoutExtension($Instance)
$TmpSol = "$env:TEMP\hgs_bench_openmp.sol"

function Run-Config {
    param([string]$Label, [int]$OpenMpFlag, [int]$NumThreads)

    $results = @()
    $env:OMP_NUM_THREADS = $NumThreads

    for ($i = 1; $i -le $Runs; $i++) {
        $runSeed = $Seed + $i - 1
        $args = @($Instance, $TmpSol, "-seed", $runSeed, "-t", $TimeLimit, "-openMP", $OpenMpFlag, "-log", 0)

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $ExePath @args 2>&1
        $sw.Stop()

        $nbIter    = $null
        $timeSpent = $null
        $finishedLine = $output | Where-Object { $_ -match "GENETIC ALGORITHM FINISHED" } | Select-Object -Last 1
        if ($finishedLine -match "FINISHED AFTER\s+(\d+)\s+ITERATIONS") { $nbIter    = [int]$matches[1] }
        if ($finishedLine -match "TIME SPENT:\s*([\d.]+)")              { $timeSpent = [double]$matches[1] }

        $bestCost = $null
        $pgFile = "$TmpSol.PG.csv"
        if (Test-Path $pgFile) {
            $lastLine = Get-Content $pgFile | Select-Object -Last 1
            if ($lastLine -match "^[^;]+;\d+;([\d.]+)") { $bestCost = [double]$matches[1] }
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
        Write-Host ("  [{0,-12}] run {1}/{2}  seed={3}  best={4}  iters={5}  time={6}" -f $Label, $i, $Runs, $runSeed, $costStr, $iterStr, $timeStr)
    }

    $env:OMP_NUM_THREADS = $null
    return $results
}

function Summarize {
    param([object[]]$Results, [string]$Label)

    $costs = $Results | Where-Object { $null -ne $_.BestCost } | ForEach-Object { $_.BestCost }
    $times = $Results | Where-Object { $null -ne $_.TimeSec  } | ForEach-Object { $_.TimeSec  }
    $iters = $Results | Where-Object { $null -ne $_.Iters    } | ForEach-Object { $_.Iters    }

    return [PSCustomObject]@{
        Label   = $Label
        Runs    = $Results.Count
        AvgCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Average).Average } else { $null }
        MinCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Minimum).Minimum } else { $null }
        AvgTime = if ($times.Count -gt 0) { ($times | Measure-Object -Average).Average } else { $null }
        AvgIter = if ($iters.Count -gt 0) { ($iters | Measure-Object -Average).Average } else { $null }
    }
}

# ── Header ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  OpenMP SWAP* Benchmark  |  instance: {0}" -f $InstanceName) -ForegroundColor Cyan
Write-Host ("  {0} runs x {1}s  |  seed base: {2}  |  threads: {3}" -f $Runs, $TimeLimit, $Seed, ($Threads -join ",")) -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$allSummaries = @()

# ── Baseline (sequential SWAP*) ───────────────────────────────────────────────
Write-Host "--- Baseline (sequential, openMP=0) ---" -ForegroundColor Yellow
$baseResults = Run-Config -Label "CPU-seq" -OpenMpFlag 0 -NumThreads 1
$allSummaries += Summarize -Results $baseResults -Label "CPU-seq (1T)"
Write-Host ""

# ── OpenMP runs at each thread count ─────────────────────────────────────────
foreach ($t in $Threads) {
    Write-Host ("--- OpenMP SWAP* ({0} threads) ---" -f $t) -ForegroundColor Green
    $res = Run-Config -Label "OMP-$t" -OpenMpFlag 1 -NumThreads $t
    $allSummaries += Summarize -Results $res -Label "OMP ($t T)"
    Write-Host ""
}

# ── Summary table ─────────────────────────────────────────────────────────────
$baseTime = $allSummaries[0].AvgTime

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$fmt = "{0,-16} | {1,10} | {2,10} | {3,10} | {4,10} | {5,8}"
Write-Host ($fmt -f "Config", "Avg Cost", "Min Cost", "Avg Time(s)", "Avg Iters", "Speedup")
Write-Host ("-" * 75)

foreach ($s in $allSummaries) {
    $speedup = if ($null -ne $baseTime -and $null -ne $s.AvgTime -and $s.AvgTime -gt 0) {
        "{0:F2}x" -f ($baseTime / $s.AvgTime)
    } else { "N/A" }

    $avgCostStr = if ($null -ne $s.AvgCost) { "{0:F2}" -f $s.AvgCost } else { "N/A" }
    $minCostStr = if ($null -ne $s.MinCost) { "{0:F2}" -f $s.MinCost } else { "N/A" }
    $avgTimeStr = if ($null -ne $s.AvgTime) { "{0:F2}" -f $s.AvgTime } else { "N/A" }
    $avgIterStr = if ($null -ne $s.AvgIter) { "{0:F0}" -f $s.AvgIter } else { "N/A" }
    Write-Host ($fmt -f $s.Label, $avgCostStr, $minCostStr, $avgTimeStr, $avgIterStr, $speedup)
}

Write-Host ""
Write-Host "  Note: speedup = baseline_avg_time / config_avg_time (wall-clock)." -ForegroundColor DarkGray
Write-Host "  Cost should stay within ~0.1% of baseline; large differences indicate" -ForegroundColor DarkGray
Write-Host "  a correctness regression, not just noise." -ForegroundColor DarkGray
Write-Host ""

Remove-Item $TmpSol -ErrorAction SilentlyContinue
