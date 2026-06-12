# smoke-test.ps1
# Runs HGS-CVRP with every meaningful feature combination and reports PASS/FAIL.
# Each run gets 10 seconds.  Use this before queuing large cluster jobs to catch
# crashes, hangs, or silent failures (no solution produced).
#
# PASS criteria per run:
#   1. Process exits with code 0
#   2. Output contains no "EXCEPTION"
#   3. Solution file was written
#   4. A valid (positive) cost was parsed from the PG.csv progress file
#
# Usage:
#   .\smoke-test.ps1 -ExePath <path\to\hgs.exe> [options]
#
# Options:
#   -ExePath      Path to the hgs executable (required)
#   -Instance     .vrp instance to use (default: X-n200-k36.vrp — small but has 36 routes,
#                 good for exercising GPU/OpenMP pair evaluation)
#   -TimeLimit    Seconds per run (default: 10)
#   -Seed         RNG seed (default: 1)
#   -MpiExec      MPI launcher name/path (default: mpiexec)

param(
    [Parameter(Mandatory=$true)]
    [string]$ExePath,

    [string]$Instance = "$PSScriptRoot\..\Instances\CVRP\X-n200-k36.vrp",
    [double]$TimeLimit = 10,
    [int]$Seed = 1,
    [string]$MpiExec = "mpiexec"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ExePath))  { Write-Error "Executable not found: $ExePath"; exit 1 }
if (-not (Test-Path $Instance)) { Write-Error "Instance not found: $Instance"; exit 1 }

$InstanceName = [System.IO.Path]::GetFileNameWithoutExtension($Instance)
$mpiAvailable = $null -ne (Get-Command $MpiExec -ErrorAction SilentlyContinue)

# ── Test catalogue ─────────────────────────────────────────────────────────────
# Each entry: Label, ExtraArgs (array), NMpiProcs (0 = no MPI)
# Note: GPU takes priority over OpenMP in the if-else chain in LocalSearch.cpp,
# so "gpu+openmp" effectively runs GPU only — included to verify it doesn't crash.

$tests = [System.Collections.Generic.List[PSCustomObject]]::new()

$tests.Add([PSCustomObject]@{
    Label    = "baseline"
    Args     = @()
    MpiProcs = 0
})
$tests.Add([PSCustomObject]@{
    Label    = "gpu"
    Args     = @("-gpu", 1)
    MpiProcs = 0
})
$tests.Add([PSCustomObject]@{
    Label    = "openmp (2T)"
    Args     = @("-openMP", 1)
    MpiProcs = 0
    Env      = @{ OMP_NUM_THREADS = "2" }
})
$tests.Add([PSCustomObject]@{
    Label    = "openmp (4T)"
    Args     = @("-openMP", 1)
    MpiProcs = 0
    Env      = @{ OMP_NUM_THREADS = "4" }
})
$tests.Add([PSCustomObject]@{
    Label    = "offspring (4:2)"
    Args     = @("-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
    MpiProcs = 0
})
$tests.Add([PSCustomObject]@{
    Label    = "offspring (8:4)"
    Args     = @("-makeManyOffspring", 1, "-numOffspring", 8, "-numThreadsOffspring", 4)
    MpiProcs = 0
})
$tests.Add([PSCustomObject]@{
    Label    = "gpu + offspring"
    Args     = @("-gpu", 1, "-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
    MpiProcs = 0
})
$tests.Add([PSCustomObject]@{
    Label    = "openmp + offspring"
    Args     = @("-openMP", 1, "-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
    MpiProcs = 0
    Env      = @{ OMP_NUM_THREADS = "2" }
})
$tests.Add([PSCustomObject]@{
    Label    = "gpu + openmp (gpu wins)"
    Args     = @("-gpu", 1, "-openMP", 1)
    MpiProcs = 0
})

# MPI island variants — only added if mpiexec is available
if ($mpiAvailable) {
    $tests.Add([PSCustomObject]@{
        Label    = "island 2 ring fixed"
        Args     = @("-island", 1, "-nbNodes", 2, "-topology", 0,
                     "-migrationPolicy", 0, "-interval", 20,
                     "-selectionCount", 1, "-immigrantHandler", 0)
        MpiProcs = 2
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 4 ring improvement"
        Args     = @("-island", 1, "-nbNodes", 4, "-topology", 0,
                     "-migrationPolicy", 1, "-warmup", 10, "-sendCooldown", 10,
                     "-receiveStagnationThreshold", 20,
                     "-selectionCount", 1, "-immigrantHandler", 1)
        MpiProcs = 4
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 4 star diversity"
        Args     = @("-island", 1, "-nbNodes", 4, "-topology", 3,
                     "-migrationPolicy", 3, "-sendCooldown", 10,
                     "-minReceiveInterval", 10, "-maxReceiveInterval", 50, "-warmup", 5,
                     "-selectionCount", 1, "-immigrantHandler", 0)
        MpiProcs = 4
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 4 randomreg adaptive"
        Args     = @("-island", 1, "-nbNodes", 4, "-topology", 5, "-topologyDegree", 2,
                     "-migrationPolicy", 2, "-sendCooldown", 10,
                     "-minReceiveInterval", 10, "-maxReceiveInterval", 50, "-warmup", 5,
                     "-selectionCount", 1, "-immigrantHandler", 2)
        MpiProcs = 4
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 2 + offspring"
        Args     = @("-island", 1, "-nbNodes", 2, "-topology", 0,
                     "-migrationPolicy", 0, "-interval", 20,
                     "-selectionCount", 1, "-immigrantHandler", 0,
                     "-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
        MpiProcs = 2
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 2 + openmp"
        Args     = @("-island", 1, "-nbNodes", 2, "-topology", 0,
                     "-migrationPolicy", 0, "-interval", 20,
                     "-selectionCount", 1, "-immigrantHandler", 0,
                     "-openMP", 1)
        MpiProcs = 2
        Env      = @{ OMP_NUM_THREADS = "2" }
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 2 + gpu"
        Args     = @("-island", 1, "-nbNodes", 2, "-topology", 0,
                     "-migrationPolicy", 0, "-interval", 20,
                     "-selectionCount", 1, "-immigrantHandler", 0,
                     "-gpu", 1)
        MpiProcs = 2
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 2 + gpu + offspring"
        Args     = @("-island", 1, "-nbNodes", 2, "-topology", 0,
                     "-migrationPolicy", 0, "-interval", 20,
                     "-selectionCount", 1, "-immigrantHandler", 0,
                     "-gpu", 1,
                     "-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
        MpiProcs = 2
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 4 + gpu + offspring"
        Args     = @("-island", 1, "-nbNodes", 4, "-topology", 0,
                     "-migrationPolicy", 1, "-warmup", 10, "-sendCooldown", 10,
                     "-receiveStagnationThreshold", 20,
                     "-selectionCount", 1, "-immigrantHandler", 1,
                     "-gpu", 1,
                     "-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
        MpiProcs = 4
    })
    $tests.Add([PSCustomObject]@{
        Label    = "island 4 star + gpu + offspring"
        Args     = @("-island", 1, "-nbNodes", 4, "-topology", 3,
                     "-migrationPolicy", 3, "-sendCooldown", 10,
                     "-minReceiveInterval", 10, "-maxReceiveInterval", 50, "-warmup", 5,
                     "-selectionCount", 1, "-immigrantHandler", 0,
                     "-gpu", 1,
                     "-makeManyOffspring", 1, "-numOffspring", 4, "-numThreadsOffspring", 2)
        MpiProcs = 4
    })
} else {
    Write-Host "  (MPI not found — island model tests will be skipped)" -ForegroundColor DarkGray
}

# ── Runner ─────────────────────────────────────────────────────────────────────
function Run-Test {
    param([PSCustomObject]$Test)

    $sol    = "$env:TEMP\hgs_smoke_$([System.IO.Path]::GetRandomFileName()).sol"
    $pgFile = "$sol.PG.csv"

    $baseArgs = @($Instance, $sol, "-seed", $Seed, "-t", $TimeLimit, "-log", 0)
    $allArgs  = $baseArgs + $Test.Args

    # Apply env overrides
    $savedEnv = @{}
    if ($Test.PSObject.Properties['Env']) {
        foreach ($kv in $Test.Env.GetEnumerator()) {
            $savedEnv[$kv.Key] = [System.Environment]::GetEnvironmentVariable($kv.Key)
            [System.Environment]::SetEnvironmentVariable($kv.Key, $kv.Value)
        }
    }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        if ($Test.MpiProcs -gt 0) {
            $output = & $MpiExec -n $Test.MpiProcs $ExePath @allArgs 2>&1
        } else {
            $output = & $ExePath @allArgs 2>&1
        }
        $exitCode = $LASTEXITCODE
    } catch {
        $output   = $_.Exception.Message
        $exitCode = -1
    }
    $sw.Stop()

    # Restore env
    foreach ($kv in $savedEnv.GetEnumerator()) {
        [System.Environment]::SetEnvironmentVariable($kv.Key, $kv.Value)
    }

    # ── Checks ──
    $failures = [System.Collections.Generic.List[string]]::new()

    if ($exitCode -ne 0) {
        $failures.Add("exit code $exitCode")
    }

    $exceptionLines = $output | Where-Object { $_ -match "EXCEPTION" }
    if ($exceptionLines) {
        $msg = ($exceptionLines | Select-Object -First 1).ToString().Trim()
        $failures.Add("exception: $msg")
    }

    if (-not (Test-Path $sol)) {
        $failures.Add("no solution file written")
    } else {
        Remove-Item $sol -ErrorAction SilentlyContinue
    }

    $bestCost = $null
    if (Test-Path $pgFile) {
        $lastLine = Get-Content $pgFile | Select-Object -Last 1
        if ($lastLine -match "^[^;]+;\d+;([\d.]+)") {
            $bestCost = [double]$matches[1]
            if ($bestCost -le 0) { $failures.Add("cost <= 0 ($bestCost)") }
        } else {
            $failures.Add("PG.csv unreadable: '$lastLine'")
        }
        Remove-Item $pgFile -ErrorAction SilentlyContinue
    } else {
        $failures.Add("no PG.csv written")
    }

    $pass    = $failures.Count -eq 0
    $costStr = if ($null -ne $bestCost) { "{0:F2}" -f $bestCost } else { "N/A" }
    $wallStr = "{0:F1}s" -f $sw.Elapsed.TotalSeconds
    if ($pass) {
        $detail = "cost=$costStr  wall=$wallStr"
    } else {
        $detail = $failures -join "; "
        if ($null -ne $bestCost) { $detail += "  (best=$costStr)" }
    }

    return [PSCustomObject]@{
        Label    = $Test.Label
        Pass     = $pass
        Detail   = $detail
        BestCost = $bestCost
        WallSec  = $sw.Elapsed.TotalSeconds
    }
}

# ── Main ───────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ("  HGS-CVRP Smoke Test  |  instance: {0}" -f $InstanceName) -ForegroundColor Cyan
Write-Host ("  {0} tests  |  {1}s each  |  seed: {2}" -f $tests.Count, $TimeLimit, $Seed) -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$results = @()
$padLabel = ($tests | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
$padLabel = [Math]::Max($padLabel, 10)

foreach ($test in $tests) {
    Write-Host ("  Running: {0} ..." -f $test.Label.PadRight($padLabel)) -NoNewline
    $r = Run-Test -Test $test
    $results += $r

    $statusColor = if ($r.Pass) { "Green" } else { "Red" }
    $statusText  = if ($r.Pass) { "PASS" }  else { "FAIL" }
    Write-Host (" [{0}]  {1}" -f $statusText, $r.Detail) -ForegroundColor $statusColor
}

# ── Summary ────────────────────────────────────────────────────────────────────
$passed = ($results | Where-Object { $_.Pass }).Count
$failed = ($results | Where-Object { -not $_.Pass }).Count
$total  = $results.Count

$padLabel = ($results | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum
$padLabel = [Math]::Max($padLabel, 10)

Write-Host ""
Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

$header = "  {0}  {1}  {2}  {3}" -f "Test".PadRight($padLabel), "Status", "BestCost".PadLeft(12), "Wall".PadLeft(7)
Write-Host $header -ForegroundColor Cyan
Write-Host ("  " + ("-" * ($padLabel + 36))) -ForegroundColor DarkGray

foreach ($r in $results) {
    $statusColor = if ($r.Pass) { "Green" } else { "Red" }
    $statusText  = if ($r.Pass) { "PASS  " } else { "FAIL  " }
    $costCol     = if ($null -ne $r.BestCost) { ("{0:F2}" -f $r.BestCost).PadLeft(12) } else { "N/A".PadLeft(12) }
    $wallCol     = ("{0:F1}s" -f $r.WallSec).PadLeft(7)
    $line = "  {0}  {1}  {2}  {3}" -f $r.Label.PadRight($padLabel), $statusText, $costCol, $wallCol
    Write-Host $line -ForegroundColor $statusColor
}

Write-Host ("  " + ("-" * ($padLabel + 36))) -ForegroundColor DarkGray
$summaryColor = if ($failed -eq 0) { "Green" } else { "Red" }
Write-Host ("  RESULT: {0}/{1} passed" -f $passed, $total) -ForegroundColor $summaryColor

if ($failed -gt 0) {
    Write-Host ""
    Write-Host "  FAILED details:" -ForegroundColor Red
    foreach ($r in ($results | Where-Object { -not $_.Pass })) {
        Write-Host ("    - {0}: {1}" -f $r.Label, $r.Detail) -ForegroundColor Red
    }
}
Write-Host "══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

exit $failed
