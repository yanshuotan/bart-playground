$scriptDir = $PSScriptRoot
$repoDir = (Resolve-Path (Join-Path $scriptDir "..\..")).Path

$python = Join-Path $repoDir ".venv\Scripts\python.exe"
$script = Join-Path $scriptDir "run_all_analyses.py"
$log = Join-Path $scriptDir "analysis_outputs\all_datasets.log"

$datasets = @(
    "fixed100_CalHousing_subsample5000",
    # "fixed100_CCPP",
    "fixed100_Concrete",
    "fixed100_Friedman",
    # "fixed100_FriedmanSparseDir_p20",
    # "fixed100_FriedmanSparseDir_p100",
    # "fixed100_FriedmanSparseDir_p200",
    "fixed100_SeoulBike"
)

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

& {
    foreach ($dataset in $datasets) {
        Write-Host "`n===== Running $dataset ====="

        try {
            & $python $script $dataset 2>&1
            $exitCode = $LASTEXITCODE

            if ($exitCode -ne 0) {
                Write-Warning "$dataset failed with exit code $exitCode; continuing."
            }
            else {
                Write-Host "===== Completed $dataset ====="
            }
        }
        catch {
            Write-Warning "$dataset failed: $($_.Exception.Message); continuing."
        }
    }

    Write-Host "`nAll datasets have been processed."
} *>&1 | Tee-Object -FilePath $log
