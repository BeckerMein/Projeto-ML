param(
    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [switch]$SkipBaseline,
    [switch]$SkipRandomForest,
    [switch]$SkipMlp,
    [switch]$ShutdownOnSuccess,
    [int]$ShutdownDelaySeconds = 300
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$RunRoot = Join-Path $ProjectRoot "runs\$RunId"
$NotebookRunDir = Join-Path $RunRoot "notebooks"
$ArtifactsDir = Join-Path $ProjectRoot "artifacts\modeling\$RunId"
$LogsDir = Join-Path $ArtifactsDir "logs"
$MlflowDir = Join-Path $ArtifactsDir "mlruns"
$MlflowDbPath = Join-Path $ArtifactsDir "mlflow.db"

New-Item -ItemType Directory -Force -Path $NotebookRunDir, $LogsDir, $MlflowDir | Out-Null

$env:MODEL_RUN_ID = $RunId
$env:MODEL_ARTIFACTS_DIR = $ArtifactsDir
$env:MLFLOW_ARTIFACT_ROOT_DIR = $MlflowDir
$env:MLFLOW_TRACKING_URI = "sqlite:///" + ([System.IO.Path]::GetFullPath($MlflowDbPath)).Replace("\", "/")
$env:MLFLOW_ARTIFACT_ROOT = ([System.Uri](Resolve-Path -LiteralPath $MlflowDir).Path).AbsoluteUri

$PythonResolved = Resolve-Path -LiteralPath $PythonPath -ErrorAction SilentlyContinue
if (-not $PythonResolved) {
    throw "Python nao encontrado em '$PythonPath'. Ative/crie a venv e rode: pip install -r requirements.txt"
}
$PythonExe = $PythonResolved.Path

$GoldDaily = Join-Path $ProjectRoot "data\gold\inmet_pe_daily.csv"
if (-not (Test-Path -LiteralPath $GoldDaily)) {
    throw "Tabela gold nao encontrada em '$GoldDaily'. Gere a camada gold antes do treino."
}

Write-Host "Validando nbconvert..." -ForegroundColor Cyan
& $PythonExe -m jupyter nbconvert --version | Tee-Object -FilePath (Join-Path $LogsDir "nbconvert-version.log")
if ($LASTEXITCODE -ne 0) {
    throw "nbconvert nao esta disponivel. Rode: pip install -r requirements.txt"
}

function Invoke-NotebookTraining {
    param(
        [Parameter(Mandatory = $true)][string]$NotebookPath,
        [Parameter(Mandatory = $true)][string]$OutputName,
        [Parameter(Mandatory = $true)][string]$LogName
    )

    $NotebookFullPath = Join-Path $ProjectRoot $NotebookPath
    if (-not (Test-Path -LiteralPath $NotebookFullPath)) {
        throw "Notebook nao encontrado: $NotebookFullPath"
    }

    $LogPath = Join-Path $LogsDir $LogName
    $Start = Get-Date
    Write-Host ""
    Write-Host "[$($Start.ToString('yyyy-MM-dd HH:mm:ss'))] Iniciando $NotebookPath" -ForegroundColor Cyan
    Write-Host "Log: $LogPath" -ForegroundColor DarkGray

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $PythonExe -m jupyter nbconvert `
            --to notebook `
            --execute $NotebookFullPath `
            --output $OutputName `
            --output-dir $NotebookRunDir `
            --ExecutePreprocessor.timeout=-1 `
            --ExecutePreprocessor.kernel_name=python3 `
            2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogPath

        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    $End = Get-Date
    $Duration = New-TimeSpan -Start $Start -End $End

    if ($ExitCode -ne 0) {
        Write-Host "[$($End.ToString('yyyy-MM-dd HH:mm:ss'))] Falha em $NotebookPath depois de $($Duration.ToString())." -ForegroundColor Red
        Write-Host "Consulte o log: $LogPath" -ForegroundColor Yellow
        throw "Execucao interrompida no notebook $NotebookPath."
    }

    Write-Host "[$($End.ToString('yyyy-MM-dd HH:mm:ss'))] Concluido $NotebookPath em $($Duration.ToString())." -ForegroundColor Green
    Write-Host "Notebook executado com resultados: $(Join-Path $NotebookRunDir $OutputName)" -ForegroundColor Green
}

Write-Host "Raiz do projeto: $ProjectRoot" -ForegroundColor Cyan
Write-Host "Diretorio dos notebooks executados: $NotebookRunDir" -ForegroundColor Cyan
Write-Host "Diretorio dos artefatos: $ArtifactsDir" -ForegroundColor Cyan
Write-Host "Os logs serao exibidos no console e salvos em artifacts\modeling\$RunId\logs." -ForegroundColor Cyan

if (-not $SkipBaseline) {
    Invoke-NotebookTraining `
        -NotebookPath "notebooks\modelos\00_baseline_climatologica.ipynb" `
        -OutputName "00_baseline_climatologica.executed.ipynb" `
        -LogName "00_baseline_climatologica.log"
}

if (-not $SkipRandomForest) {
    Invoke-NotebookTraining `
        -NotebookPath "notebooks\modelos\01_random_forest.ipynb" `
        -OutputName "01_random_forest.executed.ipynb" `
        -LogName "01_random_forest.log"
}

if (-not $SkipMlp) {
    Invoke-NotebookTraining `
        -NotebookPath "notebooks\modelos\02_mlp.ipynb" `
        -OutputName "02_mlp.executed.ipynb" `
        -LogName "02_mlp.log"
}

Write-Host ""
Write-Host "Treinamentos finalizados com sucesso." -ForegroundColor Green
Write-Host "Notebooks executados: $NotebookRunDir" -ForegroundColor Green
Write-Host "Artefatos da execucao: $ArtifactsDir" -ForegroundColor Green

if ($ShutdownOnSuccess) {
    Write-Host "Agendando shutdown em $ShutdownDelaySeconds segundos. Para cancelar: shutdown /a" -ForegroundColor Yellow
    shutdown /s /t $ShutdownDelaySeconds /c "Treinamentos finalizados com sucesso. Desligando em $ShutdownDelaySeconds segundos."
}
