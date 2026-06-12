# benchmark-scaling.ps1
# Runs a chosen configuration across a range of instance sizes to show how
# wall-clock time and solution quality scale with problem size (n).
#
# Useful for spotting super-linear complexity growth or sudden performance
# cliffs (e.g., the OpenMP threadBestInsert OOM issue for large instances).
#
# Usage:
#   .\benchmark-scaling.ps1 -ExePath <path\to\hgs.exe> [options]
#
# Options:
#   -ExePath      Path to the hgs executable (required)
#   -InstanceDir  Directory containing .vrp files (default: Instances\CVRP)
#   -Seed         Fixed seed (default: 1)
#   -TimeLimit    Seconds per run (default: 10)
#   -Runs         Repeated runs per instance (default: 2)
#   -Mode         "cpu" | "openmp" | "gpu" | "offspring" (default: "cpu")
#   -Threads      OMP_NUM_THREADS when Mode=openmp (default: 4)
#   -Instances    Explicit list of .vrp basenames to run; overrides auto-select

param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [string]$InstanceDir = "$PSScriptRoot\..\Instances\CVRP",
    [int]$Seed = 1,
    [double]$TimeLimit = 10,
    [int]$Runs = 2,
    [ValidateSet("cpu","openmp","gpu","offspring")]
    [string]$Mode = "cpu",
    [int]$Threads = 4,
    [string[]]$Instances = @()
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) { Write-Error "Executable not found: $ExePath"; exit 1 }
if (-not (Test-Path $InstanceDir)) { Write-Error "Instance directory not found: $InstanceDir"; exit 1 }

# Default set: representative spread across small → large (X-series only)
$defaultInstances = @(
    "X-n101-k25",
    "X-n157-k13",
    "X-n261-k13",
    "X-n401-k29",
    "X-n502-k39",
    "X-n627-k43",
    "X-n749-k98",
    "X-n876-k59",
    "X-n1001-k43"
)

if ($Instances.Count -eq 0) { $Instances = $defaultInstances }

$TmpSol = "$env:TEMP\hgs_bench_scaling.sol"

function Build-Args {
    param([string]$Mode, [int]$Threads)
    switch ($Mode) {
        "cpu"      { return @() }
        "openmp"   { return @("-openMP", 1) }
        "gpu"      { return @("-gpu", 1) }
        "offspring"{ return @("-makeManyOffspring", 1, "-numOffspring", $Threads, "-numThreadsOffspring", $Threads) }
    }
    return @()
}

function Run-Instance {
    param([string]$InstPath, [string[]]$ModeArgs)

    $costs = @()
    $walls = @()
    $iters = @()

    for ($i = 1; $i -le $Runs; $i++) {
        $runSeed = $Seed + $i - 1
        $args = @($InstPath, $TmpSol, "-seed", $runSeed, "-t", $TimeLimit, "-log", 0) + $ModeArgs

        if ($Mode -eq "openmp") { $env:OMP_NUM_THREADS = $Threads }

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $ExePath @args 2>&1
        $sw.Stop()
        $wallSec = $sw.Elapsed.TotalSeconds

        if ($Mode -eq "openmp") { $env:OMP_NUM_THREADS = $null }

        $nbIter = $null
        $finishedLine = $output | Where-Object { $_ -match "GENETIC ALGORITHM FINISHED" } | Select-Object -Last 1
        if ($finishedLine -match "FINISHED AFTER\s+(\d+)\s+ITERATIONS") { $nbIter = [int]$matches[1] }

        $bestCost = $null
        $pgFile = "$TmpSol.PG.csv"
        if (Test-Path $pgFile) {
            $lastLine = Get-Content $pgFile | Select-Object -Last 1
            if ($lastLine -match "^[^;]+;\d+;([\d.]+)") { $bestCost = [double]$matches[1] }
            Remove-Item $pgFile -ErrorAction SilentlyContinue
        }

        if ($null -ne $bestCost) { $costs += $bestCost }
        $walls += $wallSec
        if ($null -ne $nbIter) { $iters += $nbIter }
    }

    return [PSCustomObject]@{
        AvgCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Average).Average } else { $null }
        MinCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Minimum).Minimum } else { $null }
        AvgWall = if ($walls.Count -gt 0) { ($walls | Measure-Object -Average).Average } else { $null }
        AvgIter = if ($iters.Count -gt 0) { ($iters | Measure-Object -Average).Average } else { $null }
    }
}

# ── Header ─────────────────────────────────────────────────────────────────────
$modeLabel = switch ($Mode) {
    "cpu"      { "sequential CPU" }
    "openmp"   { "OpenMP SWAP* ($Threads threads)" }
    "gpu"      { "GPU SWAP*" }
    "offspring"{ "parallel offspring ($Threads threads)" }
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  Scaling Benchmark  |  mode: {0}" -f $modeLabel) -ForegroundColor Cyan
Write-Host ("  {0} runs x {1}s  |  seed base: {2}  |  {3} instances" -f $Runs, $TimeLimit, $Seed, $Instances.Count) -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$modeArgs = Build-Args -Mode $Mode -Threads $Threads
$rows = @()

$fmt = "{0,-22} | {1,7} | {2,10} | {3,10} | {4,10} | {5,10}"
Write-Host ($fmt -f "Instance", "n", "Avg Cost", "Min Cost", "Avg Wall(s)", "Avg Iters")
Write-Host ("-" * 79)

foreach ($instName in $Instances) {
    # Accept with or without .vrp extension
    $instBase = $instName -replace "\.vrp$", ""
    $instPath = Join-Path $InstanceDir "$instBase.vrp"

    if (-not (Test-Path $instPath)) {
        Write-Warning "  Skipping '$instBase' — file not found at $instPath"
        continue
    }

    # Extract n from filename (X-n<N>-k<K>)
    $n = $null
    if ($instBase -match "-n(\d+)-") { $n = [int]$matches[1] }

    Write-Host ("  Running {0} ..." -f $instBase) -ForegroundColor DarkGray
    $r = Run-Instance -InstPath $instPath -ModeArgs $modeArgs

    $avgCostStr = if ($null -ne $r.AvgCost) { "{0:F2}" -f $r.AvgCost } else { "N/A" }
    $minCostStr = if ($null -ne $r.MinCost) { "{0:F2}" -f $r.MinCost } else { "N/A" }
    $avgWallStr = if ($null -ne $r.AvgWall) { "{0:F2}" -f $r.AvgWall } else { "N/A" }
    $avgIterStr = if ($null -ne $r.AvgIter) { "{0:F0}" -f $r.AvgIter } else { "N/A" }
    $nStr       = if ($null -ne $n)         { "$n" }                   else { "?" }

    Write-Host ($fmt -f $instBase, $nStr, $avgCostStr, $minCostStr, $avgWallStr, $avgIterStr)

    $rows += [PSCustomObject]@{
        Instance = $instBase
        N        = $n
        AvgCost  = $r.AvgCost
        MinCost  = $r.MinCost
        AvgWall  = $r.AvgWall
        AvgIter  = $r.AvgIter
    }
}

Write-Host ""
Write-Host ("  Mode: {0}  |  Time limit: {1}s per run" -f $modeLabel, $TimeLimit) -ForegroundColor DarkGray
Write-Host "  Tip: re-run with -Mode openmp -Threads 8 and compare Avg Wall(s) column." -ForegroundColor DarkGray
Write-Host ""

# Optional CSV export
$csvPath = "$PSScriptRoot\scaling_results_${Mode}.csv"
$rows | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host ("  Results saved to: {0}" -f $csvPath) -ForegroundColor DarkGray
Write-Host ""

Remove-Item $TmpSol -ErrorAction SilentlyContinue
