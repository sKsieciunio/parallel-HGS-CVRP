<#
.SYNOPSIS
    Overnight benchmark suite — 4 instances × 6 presets, 600 s each.
.PARAMETER ExePath
    Path to hgs.exe (required).
.PARAMETER InstanceDir
    Directory containing .vrp files (default: Instances\CVRP).
.PARAMETER ResultsDir
    Root directory for all output (default: results). Never uses TEMP.
.PARAMETER Seed
    RNG seed used for every run (default: 1).
.EXAMPLE
    .\scripts\benchmark-overnight.ps1 -ExePath build\release\Release\hgs.exe
#>
param(
    [Parameter(Mandatory)]
    [string]$ExePath,

    [string]$InstanceDir = "Instances\CVRP",
    [string]$ResultsDir  = "results",
    [int]$Seed           = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Validate exe ──────────────────────────────────────────────────────────────
if (-not (Test-Path $ExePath)) { throw "hgs.exe not found at: $ExePath" }
$ExePath = (Resolve-Path $ExePath).Path

$mpiexecCmd = Get-Command mpiexec -ErrorAction SilentlyContinue
$mpiexecAvail = $null -ne $mpiexecCmd

# ── Create timestamped results directory ──────────────────────────────────────
$runTag     = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir     = Join-Path $ResultsDir $runTag
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$runDir     = (Resolve-Path $runDir).Path
$summaryLog = Join-Path $runDir "summary.log"

Write-Host ""
Write-Host "======================================================"
Write-Host "  HGS-CVRP overnight benchmark  --  $runTag"
Write-Host "  Results : $runDir"
Write-Host "======================================================"
Write-Host ""

# ── Instance list ─────────────────────────────────────────────────────────────
$instances = @(
    [pscustomobject]@{ Stem="XL-n2354-k631";  Desc="~2.4k clients,  631 veh" }
    [pscustomobject]@{ Stem="XL-n2634-k17";   Desc="~2.6k clients,   17 veh" }
    [pscustomobject]@{ Stem="XL-n5288-k1246"; Desc="~5.3k clients, 1246 veh" }
    [pscustomobject]@{ Stem="XL-n5174-k55";   Desc="~5.2k clients,   55 veh" }
)

# ── Preset list ───────────────────────────────────────────────────────────────
$presets = @(
    [pscustomobject]@{
        Name   = "baseline"
        Args   = [string[]]@()
        UseMpi = $false
        NProcs = 1
    }
    [pscustomobject]@{
        Name   = "gpu"
        Args   = [string[]]@("-gpu","1")
        UseMpi = $false
        NProcs = 1
    }
    [pscustomobject]@{
        Name   = "offspring16t8"
        Args   = [string[]]@("-makeManyOffspring","1","-numOffspring","16","-numThreadsOffspring","8")
        UseMpi = $false
        NProcs = 1
    }
    [pscustomobject]@{
        Name   = "gpu_offspring16t8"
        Args   = [string[]]@("-gpu","1","-makeManyOffspring","1","-numOffspring","16","-numThreadsOffspring","8")
        UseMpi = $false
        NProcs = 1
    }
    [pscustomobject]@{
        Name   = "island8_star"
        Args   = [string[]]@("-island","1","-nbNodes","8","-topology","3")
        UseMpi = $true
        NProcs = 8
    }
    [pscustomobject]@{
        Name   = "island8_star_gpu_off16t8"
        Args   = [string[]]@("-island","1","-nbNodes","8","-topology","3",
                             "-gpu","1",
                             "-makeManyOffspring","1","-numOffspring","16","-numThreadsOffspring","8")
        UseMpi = $true
        NProcs = 8
    }
)

# ── Helpers ───────────────────────────────────────────────────────────────────
function Format-HHmmss([int]$totalSec) {
    $h = [int]($totalSec / 3600)
    $m = [int](($totalSec % 3600) / 60)
    $s = [int]($totalSec % 60)
    "{0:D2}:{1:D2}:{2:D2}" -f $h, $m, $s
}

function Write-Log([string]$msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts]  $msg"
}

function Append-Summary([string]$line) {
    Add-Content -Path $summaryLog -Value $line
}

# ── Build job list ────────────────────────────────────────────────────────────
$jobs = [System.Collections.Generic.List[pscustomobject]]::new()
foreach ($inst in $instances) {
    $stem = $inst.Stem
    $desc = $inst.Desc
    $vrp  = Join-Path (Resolve-Path $InstanceDir).Path "$stem.vrp"
    if (-not (Test-Path $vrp)) {
        Write-Warning "Instance not found, skipping: $vrp"
        continue
    }
    foreach ($preset in $presets) {
        $jobs.Add([pscustomobject]@{
            Stem   = $stem
            Desc   = $desc
            Vrp    = $vrp
            Preset = $preset
        })
    }
}

$total      = $jobs.Count
$okCount    = 0
$failCount  = 0
$skipCount  = 0
$suiteStart = [datetime]::Now

Write-Host ("Runs planned : {0}  ({1} instances x {2} presets)" -f $total, $instances.Count, $presets.Count)
Write-Host ("Max wall time: ~{0} min  (assumes sequential, 600 s/run)" -f [int]($total * 600 / 60))
Write-Host ""

Append-Summary ("Benchmark started : " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Append-Summary ("hgs.exe           : $ExePath")
Append-Summary ("seed              : $Seed")
Append-Summary ("runs planned      : $total")
Append-Summary ""
Append-Summary "idx  status  preset                      instance          exit   wall(s)"
Append-Summary "---  ------  --------------------------  ----------------  -----  -------"

# ── Main loop ─────────────────────────────────────────────────────────────────
$runIndex = 0
foreach ($job in $jobs) {
    $runIndex++
    $preset = $job.Preset
    $stem   = $job.Stem

    # Estimate remaining time based on runs completed so far
    $completedBefore = $runIndex - 1
    $elapsedSec = ([datetime]::Now - $suiteStart).TotalSeconds
    $avgSec     = if ($completedBefore -gt 0) { $elapsedSec / $completedBefore } else { 600.0 }
    $remSec     = [int](($total - $runIndex + 1) * $avgSec)
    $etaStr     = "~" + (Format-HHmmss $remSec) + " remaining"

    # ── Skip island runs if mpiexec is absent ────────────────────────────────
    if ($preset.UseMpi -and -not $mpiexecAvail) {
        $skipCount++
        Write-Log ("SKIP  [{0,2}/{1}]  {2,-26}  {3}" -f $runIndex, $total, $preset.Name, $stem)
        Append-Summary ("{0,3}  SKIP    {1,-26}  {2,-16}  n/a    n/a    (mpiexec not in PATH)" -f $runIndex, $preset.Name, $stem)
        continue
    }

    # ── Per-run output directory ──────────────────────────────────────────────
    $outDir  = Join-Path $runDir ("{0:D2}__{1}__{2}" -f $runIndex, $stem, $preset.Name)
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
    $solPath = Join-Path $outDir "solution.sol"
    $stdOut  = Join-Path $outDir "stdout.log"
    $stdErr  = Join-Path $outDir "stderr.log"

    Write-Log ("START [{0,2}/{1}]  {2,-26}  {3}  ({4})" -f $runIndex, $total, $preset.Name, $stem, $etaStr)

    # ── Build hgs argument list ───────────────────────────────────────────────
    $hgsArgs = [string[]]@($job.Vrp, $solPath, "-t", "600", "-seed", "$Seed", "-log", "1") + $preset.Args

    $runStart = [datetime]::Now
    $exitCode = 0
    $timedOut = $false

    try {
        if ($preset.UseMpi) {
            $allArgs = [string[]]@("-n", "$($preset.NProcs)", $ExePath) + $hgsArgs
            $proc = Start-Process -FilePath "mpiexec" `
                                  -ArgumentList $allArgs `
                                  -RedirectStandardOutput $stdOut `
                                  -RedirectStandardError  $stdErr `
                                  -NoNewWindow -PassThru
        } else {
            $proc = Start-Process -FilePath $ExePath `
                                  -ArgumentList $hgsArgs `
                                  -RedirectStandardOutput $stdOut `
                                  -RedirectStandardError  $stdErr `
                                  -NoNewWindow -PassThru
        }

        # Wait up to 700 s (600 s run limit + 100 s grace); kill if still alive
        $exited = $proc.WaitForExit(700000)
        if (-not $exited) {
            $timedOut = $true
            try { $proc.Kill() } catch {}
            $exitCode = -2
        } else {
            $exitCode = $proc.ExitCode
        }
    } catch {
        $exitCode = -1
        "$_" | Add-Content -Path $stdErr
    }

    $wallSec = [int]([datetime]::Now - $runStart).TotalSeconds

    if ($timedOut) {
        $statusStr  = "TIMEOUT"
        $consoleOut = "TIMEOUT after ${wallSec}s (killed)"
        $failCount++
    } elseif ($exitCode -eq 0) {
        $statusStr  = "OK"
        $consoleOut = "done  (${wallSec}s)"
        $okCount++
    } else {
        $statusStr  = "FAIL"
        $consoleOut = "FAILED exit=$exitCode  (${wallSec}s) -- see $stdErr"
        $failCount++
    }

    Write-Log ("END   [{0,2}/{1}]  {2,-26}  {3}  => {4}" -f $runIndex, $total, $preset.Name, $stem, $consoleOut)
    Append-Summary ("{0,3}  {1,-6}  {2,-26}  {3,-16}  {4,5}   {5,7}" -f `
        $runIndex, $statusStr, $preset.Name, $stem, $exitCode, $wallSec)
}

# ── Final summary ─────────────────────────────────────────────────────────────
$suiteWallSec = [int]([datetime]::Now - $suiteStart).TotalSeconds

Append-Summary ""
Append-Summary ("Finished  : " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Append-Summary ("Wall time : " + (Format-HHmmss $suiteWallSec))
Append-Summary ("OK / FAIL / SKIP : $okCount / $failCount / $skipCount")

Write-Host ""
Write-Host "======================================================"
Write-Host ("  Suite finished in " + (Format-HHmmss $suiteWallSec))
Write-Host ("  OK / FAIL / SKIP : $okCount / $failCount / $skipCount")
Write-Host "  Results   : $runDir"
Write-Host "  Summary   : $summaryLog"
Write-Host "======================================================"
Write-Host ""

# Exit with non-zero if any run failed (useful for CI or piped scripts)
if ($failCount -gt 0) { exit 1 }
