$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    & .venv/Scripts/python.exe -m ruff check src tests scripts
    if ($LASTEXITCODE) { throw 'Lint failed.' }
    & .venv/Scripts/python.exe -m pytest -q --basetemp "data/pytest-$([guid]::NewGuid().ToString('N'))"
    if ($LASTEXITCODE) { throw 'Tests failed.' }
} finally { Pop-Location }
