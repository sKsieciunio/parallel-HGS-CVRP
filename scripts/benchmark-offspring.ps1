# benchmark-offspring.ps1
# Compares sequential offspring production (baseline) against the parallel
# master-slave model (ParallelOffspringMaker) at different numOffspring /
# numThreads combinations.
#
# What is measured:
#   - Wall-clock time for a fixed time budget (-t)
#   - Number of GA iterations completed (proxy for offspring throughput)
#   - Best solution cost (to catch correctness regressions)
#
# Usage:
#   .\benchmark-offspring.ps1 -ExePath <path\to\hgs.exe> [options]
#
# Options:
#   -ExePath      Path to the hgs executable (required)
#   -Instance     Path to .vrp instance (default: X-n1001-k43.vrp)
#   -Seed         Fixed seed (default: 1)
#   -TimeLimit    Seconds per run (default: 10)
#   -Runs         Repeated runs per config (default: 3)
#   -Configs      Array of "offspring:threads" strings to test
#                 (default: "4:2", "8:4", "16:4", "16:8")

param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [string]$Instance = "$PSScriptRoot\..\Instances\CVRP\X-n1001-k43.vrp",
    [int]$Seed = 1,
    [double]$TimeLimit = 10,
    [int]$Runs = 3,
    [string[]]$Configs = @("4:2", "8:4", "16:4", "16:8")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) { Write-Error "Executable not found: $ExePath"; exit 1 }
if (-not (Test-Path $Instance)) { Write-Error "Instance file not found: $Instance"; exit 1 }

$InstanceName = [System.IO.Path]::GetFileNameWithoutExtension($Instance)
$TmpSol = "$env:TEMP\hgs_bench_offspring.sol"

function Run-Config {
    param([string]$Label, [string[]]$ExtraArgs)

    $results = @()

    for ($i = 1; $i -le $Runs; $i++) {
        $runSeed = $Seed + $i - 1
        $args = @($Instance, $TmpSol, "-seed", $runSeed, "-t", $TimeLimit, "-log", 0) + $ExtraArgs

        $output = & $ExePath @args 2>&1

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
        Write-Host ("  [{0,-18}] run {1}/{2}  seed={3}  best={4}  iters={5}  time={6}" -f $Label, $i, $Runs, $runSeed, $costStr, $iterStr, $timeStr)
    }

    return $results
}

function Summarize {
    param([object[]]$Results, [string]$Label)

    $costs = $Results | Where-Object { $null -ne $_.BestCost } | ForEach-Object { $_.BestCost }
    $times = $Results | Where-Object { $null -ne $_.TimeSec  } | ForEach-Object { $_.TimeSec  }
    $iters = $Results | Where-Object { $null -ne $_.Iters    } | ForEach-Object { $_.Iters    }

    return [PSCustomObject]@{
        Label   = $Label
        AvgCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Average).Average } else { $null }
        MinCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Minimum).Minimum } else { $null }
        AvgTime = if ($times.Count -gt 0) { ($times | Measure-Object -Average).Average } else { $null }
        AvgIter = if ($iters.Count -gt 0) { ($iters | Measure-Object -Average).Average } else { $null }
    }
}

# ── Header ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  Parallel Offspring Benchmark  |  instance: {0}" -f $InstanceName) -ForegroundColor Cyan
Write-Host ("  {0} runs x {1}s  |  seed base: {2}" -f $Runs, $TimeLimit, $Seed) -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$allSummaries = @()

# ── Baseline: standard sequential offspring ───────────────────────────────────
Write-Host "--- Baseline (makeManyOffspring=0) ---" -ForegroundColor Yellow
$baseResults = Run-Config -Label "sequential" -ExtraArgs @("-makeManyOffspring", 0)
$allSummaries += Summarize -Results $baseResults -Label "sequential"
Write-Host ""

# ── Parallel configs ───────────────────────────────────────────────────────────
foreach ($cfg in $Configs) {
    $parts = $cfg -split ":"
    if ($parts.Count -ne 2) { Write-Warning "Skipping invalid config '$cfg' (expected 'offspring:threads')"; continue }
    $nOff = [int]$parts[0]
    $nThr = [int]$parts[1]
    $label = "off=$nOff thr=$nThr"

    Write-Host ("--- Parallel offspring: {0} offspring, {1} threads ---" -f $nOff, $nThr) -ForegroundColor Green
    $res = Run-Config -Label $label -ExtraArgs @("-makeManyOffspring", 1, "-numOffspring", $nOff, "-numThreadsOffspring", $nThr)
    $allSummaries += Summarize -Results $res -Label $label
    Write-Host ""
}

# ── Summary table ──────────────────────────────────────────────────────────────
$baseIter = $allSummaries[0].AvgIter
$baseTime = $allSummaries[0].AvgTime

Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SUMMARY" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$fmt = "{0,-20} | {1,10} | {2,10} | {3,10} | {4,10} | {5,11}"
Write-Host ($fmt -f "Config", "Avg Cost", "Min Cost", "Avg Time(s)", "Avg Iters", "Iter Speedup")
Write-Host ("-" * 82)

foreach ($s in $allSummaries) {
    # Iter speedup: parallel makes N offspring per "iteration", so effective
    # throughput = avgIter * numOffspring.  We compare against baseline (1 per iter).
    $iterSpeedup = if ($null -ne $baseIter -and $null -ne $s.AvgIter -and $baseIter -gt 0) {
        "{0:F2}x" -f ($s.AvgIter / $baseIter)
    } else { "N/A" }

    $avgCostStr = if ($null -ne $s.AvgCost) { "{0:F2}" -f $s.AvgCost } else { "N/A" }
    $minCostStr = if ($null -ne $s.MinCost) { "{0:F2}" -f $s.MinCost } else { "N/A" }
    $avgTimeStr = if ($null -ne $s.AvgTime) { "{0:F2}" -f $s.AvgTime } else { "N/A" }
    $avgIterStr = if ($null -ne $s.AvgIter) { "{0:F0}" -f $s.AvgIter } else { "N/A" }
    Write-Host ($fmt -f $s.Label, $avgCostStr, $minCostStr, $avgTimeStr, $avgIterStr, $iterSpeedup)
}

Write-Host ""
Write-Host "  Note: 'Iter Speedup' = avg_iters / baseline_avg_iters." -ForegroundColor DarkGray
Write-Host "  Parallel mode does N offspring per iter so fewer iters is expected;" -ForegroundColor DarkGray
Write-Host "  watch Avg Cost — it should be <= baseline for the same time budget." -ForegroundColor DarkGray
Write-Host ""

Remove-Item $TmpSol -ErrorAction SilentlyContinue
