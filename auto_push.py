# =============================================================================
# ワンクリック Git 自動コミット＆プッシュ スクリプト
# 使い方:
#   .\auto_push.bat             (ダブルクリック、またはターミナルから実行)
#   .\auto_push.ps1             (PowerShellから実行)
#   .\auto_push.ps1 "メモ内容"   (好きなコミットメッセージを付けて実行)
# =============================================================================

from datetime import datetime
from pathlib import Path
import subprocess
import sys

# プロジェクトのルートフォルダ（このスクリプトがある場所）
REPO_DIR = Path(__file__).resolve().parent


def run_git(*args, capture=False):
    """git コマンドを実行する補助関数。"""
    return subprocess.run(
        ["git", *args],
        cwd=REPO_DIR,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main():
    # コマンドライン引数が渡されていればそれをコミットメッセージに使う
    custom_message = " ".join(sys.argv[1:]).strip()

    # 現在のブランチ名を取得（通常は main）
    branch_proc = run_git("rev-parse", "--abbrev-ref", "HEAD", capture=True)
    if branch_proc.returncode != 0:
        print("エラー: Gitリポジトリとして認識できませんでした。")
        sys.exit(1)
    branch = branch_proc.stdout.strip()

    # 変更されたファイルの一覧を取得
    status_proc = run_git("status", "--porcelain", capture=True)
    status_lines = [line for line in status_proc.stdout.splitlines() if line.strip()]

    if status_lines:
        print(f"=== 変更を検出しました ({len(status_lines)} 件) ===")
        run_git("status", "--short")

        # コミットメッセージが未指定なら「日時＋変更ファイル名」で自動生成
        if custom_message:
            message = custom_message
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_names = []
            for line in status_lines:
                raw_path = line[3:].strip().strip('"')
                file_names.append(Path(raw_path).name)
            if len(file_names) <= 3:
                summary = ", ".join(file_names)
            else:
                summary = ", ".join(file_names[:3]) + f" ほか{len(file_names) - 3}件"
            message = f"Auto push ({timestamp}): {summary}"

        print("\n1. 変更をステージング (git add -A)...")
        if run_git("add", "-A").returncode != 0:
            print("エラー: git add に失敗しました。")
            sys.exit(1)

        print(f"2. コミット作成: {message}")
        if run_git("commit", "-m", message).returncode != 0:
            print("エラー: git commit に失敗しました。")
            sys.exit(1)
    else:
        print("新しい変更ファイルはありません。")

    # まだGitHubへ送っていないコミットがあるか確認
    unpushed_proc = run_git("rev-list", "--count", "@{u}..HEAD", capture=True)
    needs_push = (
        unpushed_proc.returncode != 0
        or int(unpushed_proc.stdout.strip() or "0") > 0
    )

    if needs_push:
        print(f"\n3. GitHubへプッシュ (git push origin {branch})...")
        if run_git("push", "origin", branch).returncode != 0:
            print("エラー: git push に失敗しました。ネットワークやSSH設定を確認してください。")
            sys.exit(1)
        print(f"\n完了: GitHub ({branch}) へのプッシュに成功しました！")
    else:
        print("すでにGitHubと最新の状態です（プッシュ不要）。")


if __name__ == "__main__":
    main()
