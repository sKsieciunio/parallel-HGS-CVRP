# benchmark-island.ps1
# Compares single-island (baseline) against the MPI island model with varying
# numbers of islands, topologies, and migration policies.
#
# Requires MPI to be installed; uses mpiexec (override with -MpiExec).
# Each MPI run spawns N processes; rank 0 writes the final solution.
#
# What is measured:
#   - Best feasible cost found in the given time budget
#   - Wall-clock time (from PowerShell; the per-process CPU time logged by hgs
#     is not directly comparable across island counts)
#
# Usage:
#   .\benchmark-island.ps1 -ExePath <path\to\hgs.exe> [options]
#
# Options:
#   -ExePath        Path to the hgs executable (required)
#   -Instance       Path to .vrp instance (default: X-n1001-k43.vrp)
#   -Seed           Fixed seed base (default: 1) — each island gets seed+rank
#   -TimeLimit      Seconds per run (default: 20)
#   -Runs           Repeated runs per config (default: 3)
#   -IslandCounts   Islands to test (default: 2, 4, 8)
#   -MpiExec        Path or name of the MPI launcher (default: mpiexec)

param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [string]$Instance = "$PSScriptRoot\..\Instances\CVRP\X-n1001-k43.vrp",
    [int]$Seed = 1,
    [double]$TimeLimit = 20,
    [int]$Runs = 3,
    [int[]]$IslandCounts = @(2, 4, 8),
    [string]$MpiExec = "mpiexec"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath)) { Write-Error "Executable not found: $ExePath"; exit 1 }
if (-not (Test-Path $Instance)) { Write-Error "Instance file not found: $Instance"; exit 1 }

# Check MPI is available
$mpiAvailable = $null -ne (Get-Command $MpiExec -ErrorAction SilentlyContinue)
if (-not $mpiAvailable) {
    Write-Warning "MPI launcher '$MpiExec' not found in PATH — island runs will be skipped."
}

$InstanceName = [System.IO.Path]::GetFileNameWithoutExtension($Instance)
$TmpSol = "$env:TEMP\hgs_bench_island.sol"

function Run-Single {
    param([string]$Label, [string[]]$ExtraArgs)

    $results = @()
    for ($i = 1; $i -le $Runs; $i++) {
        $runSeed = $Seed + $i - 1
        $args = @($Instance, $TmpSol, "-seed", $runSeed, "-t", $TimeLimit, "-log", 0) + $ExtraArgs

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $ExePath @args 2>&1
        $sw.Stop()
        $wallSec = $sw.Elapsed.TotalSeconds

        $bestCost = $null
        $pgFile = "$TmpSol.PG.csv"
        if (Test-Path $pgFile) {
            $lastLine = Get-Content $pgFile | Select-Object -Last 1
            if ($lastLine -match "^[^;]+;\d+;([\d.]+)") { $bestCost = [double]$matches[1] }
            Remove-Item $pgFile -ErrorAction SilentlyContinue
        }

        $nbIter = $null
        $finishedLine = $output | Where-Object { $_ -match "GENETIC ALGORITHM FINISHED" } | Select-Object -Last 1
        if ($finishedLine -match "FINISHED AFTER\s+(\d+)\s+ITERATIONS") { $nbIter = [int]$matches[1] }

        $results += [PSCustomObject]@{ Run = $i; Seed = $runSeed; BestCost = $bestCost; WallSec = $wallSec; Iters = $nbIter }

        $costStr = if ($null -ne $bestCost) { "{0:F2}" -f $bestCost } else { "N/A" }
        $iterStr = if ($null -ne $nbIter)   { "$nbIter" }             else { "N/A" }
        Write-Host ("  [{0,-20}] run {1}/{2}  seed={3}  best={4}  iters={5}  wall={6:F2}s" -f $Label, $i, $Runs, $runSeed, $costStr, $iterStr, $wallSec)
    }
    return $results
}

function Run-Island {
    param([string]$Label, [int]$NIslands, [string[]]$ExtraArgs)

    $results = @()
    for ($i = 1; $i -le $Runs; $i++) {
        $runSeed = $Seed + $i - 1
        $hgsArgs = @($Instance, $TmpSol, "-seed", $runSeed, "-t", $TimeLimit, "-log", 0,
                     "-island", 1, "-nbNodes", $NIslands) + $ExtraArgs

        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $output = & $MpiExec -n $NIslands $ExePath @hgsArgs 2>&1
        $sw.Stop()
        $wallSec = $sw.Elapsed.TotalSeconds

        # Rank 0 writes the solution; parse PG.csv as usual
        $bestCost = $null
        $pgFile = "$TmpSol.PG.csv"
        if (Test-Path $pgFile) {
            $lastLine = Get-Content $pgFile | Select-Object -Last 1
            if ($lastLine -match "^[^;]+;\d+;([\d.]+)") { $bestCost = [double]$matches[1] }
            Remove-Item $pgFile -ErrorAction SilentlyContinue
        }

        # mpiexec merges stdout from all ranks — collect all FINISHED lines and average
        $nbIter = $null
        $iterVals = $output | Where-Object { $_ -match "FINISHED AFTER\s+(\d+)\s+ITERATIONS" } |
                    ForEach-Object { if ($_ -match "FINISHED AFTER\s+(\d+)\s+ITERATIONS") { [int]$matches[1] } }
        if ($iterVals.Count -gt 0) { $nbIter = [int](($iterVals | Measure-Object -Average).Average) }

        $results += [PSCustomObject]@{ Run = $i; Seed = $runSeed; BestCost = $bestCost; WallSec = $wallSec; Iters = $nbIter }

        $costStr = if ($null -ne $bestCost) { "{0:F2}" -f $bestCost } else { "N/A" }
        $iterStr = if ($null -ne $nbIter)   { "$nbIter (avg/island)" } else { "N/A" }
        Write-Host ("  [{0,-20}] run {1}/{2}  seed={3}  best={4}  iters={5}  wall={6:F2}s" -f $Label, $i, $Runs, $runSeed, $costStr, $iterStr, $wallSec)
    }
    return $results
}

function Summarize {
    param([object[]]$Results, [string]$Label)
    $costs  = $Results | Where-Object { $null -ne $_.BestCost } | ForEach-Object { $_.BestCost }
    $walls  = $Results | Where-Object { $null -ne $_.WallSec  } | ForEach-Object { $_.WallSec  }
    $iters  = $Results | Where-Object { $null -ne $_.Iters    } | ForEach-Object { $_.Iters    }
    return [PSCustomObject]@{
        Label   = $Label
        AvgCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Average).Average } else { $null }
        MinCost = if ($costs.Count -gt 0) { ($costs | Measure-Object -Minimum).Minimum } else { $null }
        AvgWall = if ($walls.Count -gt 0) { ($walls | Measure-Object -Average).Average } else { $null }
        AvgIter = if ($iters.Count -gt 0) { ($iters | Measure-Object -Average).Average } else { $null }
    }
}

# ── Header ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  Island Model Benchmark  |  instance: {0}" -f $InstanceName) -ForegroundColor Cyan
Write-Host ("  {0} runs x {1}s  |  seed base: {2}  |  islands: {3}" -f $Runs, $TimeLimit, $Seed, ($IslandCounts -join ",")) -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$allSummaries = @()

# ── Baseline: single island, no MPI ───────────────────────────────────────────
Write-Host "--- Baseline (1 island, no MPI) ---" -ForegroundColor Yellow
$baseResults = Run-Single -Label "1 island" -ExtraArgs @()
$allSummaries += Summarize -Results $baseResults -Label "1 island (seq)"
Write-Host ""

if (-not $mpiAvailable) {
    Write-Warning "Skipping island model runs (MPI not found)."
} else {
    # ── Island model with ring topology, fixed-interval migration ─────────────
    foreach ($n in $IslandCounts) {
        $label = "$n islands ring"
        Write-Host ("--- {0} islands, ring topology, fixed interval ---" -f $n) -ForegroundColor Green
        $res = Run-Island -Label $label -NIslands $n -ExtraArgs @(
            "-topology", 0, "-migrationPolicy", 0, "-interval", 50,
            "-selectionCount", 1, "-immigrantHandler", 0
        )
        $allSummaries += Summarize -Results $res -Label $label
        Write-Host ""
    }

    # ── 4 islands with different topologies ───────────────────────────────────
    $topoMap = @{ 0 = "ring"; 1 = "2-ring"; 3 = "star"; 4 = "full" }
    foreach ($topo in $topoMap.Keys | Sort-Object) {
        $label = "4 isl $($topoMap[$topo])"
        Write-Host ("--- 4 islands, {0} topology ---" -f $topoMap[$topo]) -ForegroundColor Magenta
        $res = Run-Island -Label $label -NIslands 4 -ExtraArgs @(
            "-topology", $topo, "-migrationPolicy", 0, "-interval", 50,
            "-selectionCount", 1, "-immigrantHandler", 0
        )
        $allSummaries += Summarize -Results $res -Label $label
        Write-Host ""
    }

    # ── 4 islands with different migration policies ───────────────────────────
    $policyMap = @{ 0 = "fixed"; 1 = "improvement"; 2 = "adaptive"; 3 = "diversity" }
    foreach ($pol in $policyMap.Keys | Sort-Object) {
        $label = "4 isl $($policyMap[$pol])"
        Write-Host ("--- 4 islands, ring, {0} migration policy ---" -f $policyMap[$pol]) -ForegroundColor DarkCyan
        $res = Run-Island -Label $label -NIslands 4 -ExtraArgs @(
            "-topology", 0, "-migrationPolicy", $pol, "-interval", 50,
            "-warmup", 20, "-sendCooldown", 20, "-receiveStagnationThreshold", 50,
            "-selectionCount", 1, "-immigrantHandler", 0
        )
        $allSummaries += Summarize -Results $res -Label $label
        Write-Host ""
    }
}

# ── Summary table ──────────────────────────────────────────────────────────────
$baseCost = $allSummaries[0].AvgCost

Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  SUMMARY  (wall-clock time includes MPI launch overhead)" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$fmt = "{0,-22} | {1,10} | {2,10} | {3,10} | {4,10} | {5,9}"
Write-Host ($fmt -f "Config", "Avg Cost", "Min Cost", "Avg Iters", "Avg Wall(s)", "Cost Gap%")
Write-Host ("-" * 83)

foreach ($s in $allSummaries) {
    $gap = if ($null -ne $baseCost -and $null -ne $s.AvgCost -and $baseCost -gt 0) {
        "{0:+0.2f}%" -f (($s.AvgCost - $baseCost) / $baseCost * 100)
    } else { "N/A" }

    $avgCostStr = if ($null -ne $s.AvgCost) { "{0:F2}" -f $s.AvgCost } else { "N/A" }
    $minCostStr = if ($null -ne $s.MinCost) { "{0:F2}" -f $s.MinCost } else { "N/A" }
    $avgIterStr = if ($null -ne $s.AvgIter) { "{0:F0}" -f $s.AvgIter } else { "N/A" }
    $avgWallStr = if ($null -ne $s.AvgWall) { "{0:F2}" -f $s.AvgWall } else { "N/A" }
    Write-Host ($fmt -f $s.Label, $avgCostStr, $minCostStr, $avgIterStr, $avgWallStr, $gap)
}

Write-Host ""
Write-Host "  Note: 'Cost Gap%' = (config_avg_cost - baseline_avg_cost) / baseline * 100." -ForegroundColor DarkGray
Write-Host "  Negative = island model found a better solution than single island." -ForegroundColor DarkGray
Write-Host "  Wall time for N islands includes MPI init; use same time budget for fair comparison." -ForegroundColor DarkGray
Write-Host ""

Remove-Item $TmpSol -ErrorAction SilentlyContinue
