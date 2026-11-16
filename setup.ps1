# Sets up RoboticFishProcessing on Windows: tools/setup_rfp.py, run with conda's own Python.
#   .\setup.ps1            set up (conda environment "rfp", git hooks, simulation meshes...)
#   .\setup.ps1 --check    only report what is missing
# If PowerShell won't run scripts: powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Find-Conda {
    $found = @($env:CONDA_EXE, (Get-Command conda -ErrorAction SilentlyContinue).Source)
    foreach ($root in "$env:USERPROFILE", "$env:LOCALAPPDATA", "C:\ProgramData") {
        foreach ($name in "miniconda3", "anaconda3", "miniforge3") {
            $found += Join-Path $root "$name\Scripts\conda.exe"
        }
    }
    # Installed somewhere else in the user's folder (e.g. Dev\Anaconda3)
    $found += Get-ChildItem $env:USERPROFILE -Directory -Depth 1 -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "conda3$|forge3$" } |
        ForEach-Object { Join-Path $_.FullName "Scripts\conda.exe" }
    foreach ($candidate in $found) {
        if ($candidate -and (Test-Path $candidate) -and $candidate -like "*.exe") { return $candidate }
    }
    return $null
}

$conda = Find-Conda
if (-not $conda) {
    Write-Host "conda isn't installed: the project's Python environment needs it."
    if ((Read-Host "Install Miniconda with winget now? [Y/n]") -in "", "y", "yes") {
        winget install --id Anaconda.Miniconda3 -e
        Write-Host "Open a new PowerShell window and run .\setup.ps1 again."
    }
    exit 1
}
$python = Join-Path (Split-Path (Split-Path $conda)) "python.exe"
& $python tools\setup_rfp.py @args
exit $LASTEXITCODE
