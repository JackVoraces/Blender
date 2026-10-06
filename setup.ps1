# Recreates everything that is not kept in Git: the Python environment, the MediaPipe
# face model and the sample portrait used by the tests.
#
#   powershell -ExecutionPolicy Bypass -File tools\setup.ps1

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root ".venv"
$ProgressPreference = 'SilentlyContinue'

if (-not (Test-Path $venv)) {
    python -m venv $venv
}
& (Join-Path $venv "Scripts\python.exe") -m pip install --disable-pip-version-check -r (Join-Path $root "requirements.txt")

$downloads = @(
    @{ Path = "models\face_landmarker.task"; Url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" },
    @{ Path = "tests\data\sample_portrait.png"; Url = "https://storage.googleapis.com/mediapipe-assets/business-person.png" }
)
foreach ($item in $downloads) {
    $target = Join-Path $root $item.Path
    if (-not (Test-Path $target)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Invoke-WebRequest -Uri $item.Url -OutFile $target
        Write-Host "downloaded $($item.Path)"
    }
}

$godot = Get-ChildItem (Join-Path $root "tools\godot") -Filter "Godot*win64.exe" -ErrorAction SilentlyContinue
if (-not $godot) {
    Write-Host "Godot is not included: put a Godot 4.x Windows build (the .exe and the _console.exe)"
    Write-Host "in tools\godot, plus an empty file named _sc_ to keep its settings inside this project."
}
