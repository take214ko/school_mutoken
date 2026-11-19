param(
    [string]$Message = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# このスクリプトがあるプロジェクトフォルダへ移動
Set-Location -Path $PSScriptRoot

# 現在のブランチ名を取得
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    Write-Host "エラー: Gitリポジトリとして認識できませんでした。" -ForegroundColor Red
    exit 1
}

# 変更ファイルがあるか確認
$statusLines = @(git status --porcelain)

if ($statusLines.Count -gt 0) {
    Write-Host "=== 変更を検出しました ($($statusLines.Count) 件) ===" -ForegroundColor Cyan
    git status --short

    # コミットメッセージが未指定なら、日時＋変更ファイル名で自動生成
    if ([string]::IsNullOrWhiteSpace($Message)) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        $fileNames = @(
            $statusLines | ForEach-Object {
                $path = $_.Substring(3).Trim().Trim('"')
                Split-Path -Path $path -Leaf
            }
        )
        if ($fileNames.Count -le 3) {
            $summary = $fileNames -join ", "
        } else {
            $summary = ($fileNames[0..2] -join ", ") + " ほか$($fileNames.Count - 3)件"
        }
        $Message = "Auto push ($timestamp): $summary"
    }

    Write-Host "`n1. 変更をステージング (git add -A)..." -ForegroundColor Yellow
    git add -A
    if ($LASTEXITCODE -ne 0) {
        Write-Host "エラー: git add に失敗しました。" -ForegroundColor Red
        exit 1
    }

    Write-Host "2. コミット作成: $Message" -ForegroundColor Yellow
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) {
        Write-Host "エラー: git commit に失敗しました。" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "新しい変更ファイルはありません。" -ForegroundColor Gray
}

# 未プッシュのコミットがあるか確認してプッシュ
$unpushedCount = (git rev-list --count "@{u}..HEAD" 2>$null)
if ($LASTEXITCODE -ne 0 -or [int]$unpushedCount -gt 0) {
    Write-Host "`n3. GitHubへプッシュ (git push origin $branch)..." -ForegroundColor Yellow
    git push origin $branch
    if ($LASTEXITCODE -ne 0) {
        Write-Host "エラー: git push に失敗しました。ネットワークやSSH設定を確認してください。" -ForegroundColor Red
        exit 1
    }
    Write-Host "`n完了: GitHub ($branch) へのプッシュに成功しました！" -ForegroundColor Green
} else {
    Write-Host "すでにGitHubと最新の状態です（プッシュ不要）。" -ForegroundColor Green
}
