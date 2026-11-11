param([switch]$DownloadModels, [switch]$RegisterCodex, [switch]$RegisterClaudeCode, [switch]$RegisterGemini)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv first: https://docs.astral.sh/uv/' }
    if (-not (Test-Path '.venv/Scripts/python.exe')) {
        uv --system-certs venv --python 3.12 .venv
        if ($LASTEXITCODE) { throw 'Python environment creation failed.' }
    }
    uv --system-certs pip install --python .venv/Scripts/python.exe 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE) { throw 'CUDA PyTorch installation failed.' }
    if (Test-Path 'requirements.lock.txt') {
        uv --system-certs pip install --python .venv/Scripts/python.exe -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
        if ($LASTEXITCODE) { throw 'Locked dependency installation failed.' }
    }
    uv --system-certs pip install --python .venv/Scripts/python.exe -e '.[inference,test]'
    if ($LASTEXITCODE) { throw 'Application installation failed.' }
    & .venv/Scripts/python.exe -m kani_tts_mcp install-cli
    if ($LASTEXITCODE) { throw 'CLI installation failed.' }
    if ($DownloadModels) {
        & .venv/Scripts/python.exe -m kani_tts_mcp setup-models
        if ($LASTEXITCODE) { throw 'Model download failed.' }
    }
    foreach ($harness in @(@{Enabled=$RegisterCodex;Name='codex'}, @{Enabled=$RegisterClaudeCode;Name='claude-code'}, @{Enabled=$RegisterGemini;Name='gemini'})) {
        if ($harness.Enabled) {
            & .venv/Scripts/python.exe -m kani_tts_mcp register $harness.Name
            if ($LASTEXITCODE) { throw "MCP registration failed: $($harness.Name)" }
        }
    }
} finally { Pop-Location }
