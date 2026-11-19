param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$MessageArgs
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
$scriptPath = Join-Path $PSScriptRoot "auto_push.py"

if (Test-Path $venvPython) {
    & $venvPython $scriptPath @MessageArgs
} else {
    & python $scriptPath @MessageArgs
}
exit $LASTEXITCODE
