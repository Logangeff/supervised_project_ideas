@echo off
setlocal EnableExtensions DisableDelayedExpansion

cd /d "%~dp0"

set "REMOTE=origin"
set "TARGET_BRANCH=main"
set "REPO_URL=https://github.com/Logangeff/supervised_project_ideas.git"
set "EXIT_CODE=1"

echo.
echo ============================================================
echo supervised_project_ideas Backup Before Push To Main
echo Git code safety only. This does NOT back up large local data files.
echo This script stages and commits current changes, backs up remote main,
echo then pushes current HEAD to remote main when safe.
echo ============================================================

git --version >nul 2>nul
if errorlevel 1 (
    echo Git is not installed or not available on PATH.
    goto :end
)

git rev-parse --git-dir >nul 2>nul
if errorlevel 1 (
    echo Initializing local git repository...
    git init
    if errorlevel 1 goto :end
)

git remote get-url %REMOTE% >nul 2>nul
if errorlevel 1 (
    git remote add %REMOTE% "%REPO_URL%"
) else (
    git remote set-url %REMOTE% "%REPO_URL%"
)

set "CURRENT_BRANCH="
for /f "usebackq delims=" %%I in (`git symbolic-ref --short -q HEAD`) do set "CURRENT_BRANCH=%%I"
if not defined CURRENT_BRANCH (
    echo Detached HEAD detected.
    echo Check out the branch you want to push, then run this script again.
    goto :end
)

set "ORIGIN_URL="
for /f "usebackq delims=" %%I in (`git remote get-url %REMOTE% 2^>nul`) do set "ORIGIN_URL=%%I"
if not defined ORIGIN_URL (
    echo Git remote "%REMOTE%" is not configured.
    goto :end
)

echo.
echo Fetching latest branches from %REMOTE%...
git fetch %REMOTE% --prune
if errorlevel 1 (
    echo Fetch failed. No backup branch or push was performed.
    goto :end
)

git rev-parse --verify "refs/remotes/%REMOTE%/%TARGET_BRANCH%" >nul 2>nul
if errorlevel 1 (
    echo %REMOTE%/%TARGET_BRANCH% does not exist.
    echo This script protects an existing remote main before pushing.
    goto :end
)

set "HEAD_SHA="
for /f "usebackq delims=" %%I in (`git rev-parse HEAD`) do set "HEAD_SHA=%%I"

set "REMOTE_MAIN_SHA="
for /f "usebackq delims=" %%I in (`git rev-parse "refs/remotes/%REMOTE%/%TARGET_BRANCH%"`) do set "REMOTE_MAIN_SHA=%%I"

echo.
echo Repo root   : %CD%
echo Remote      : %ORIGIN_URL%
echo Branch      : %CURRENT_BRANCH%
echo HEAD        : %HEAD_SHA%
echo Main remote : %REMOTE_MAIN_SHA%
echo.
echo Current status:
git status --short --branch
echo.
echo Current branches:
git branch --all --sort=refname
echo.
echo Choose a NEW backup branch name so you can recognize it later.
echo You can type a normal sentence like: before main push
echo The script will convert it to: backup/before_main_push
set /p "BACKUP_BRANCH=Backup branch name or phrase (press Enter to abort): "
if not defined BACKUP_BRANCH (
    echo.
    echo Operation canceled. Nothing was changed.
    set "EXIT_CODE=0"
    goto :end
)

set "BACKUP_BRANCH=%BACKUP_BRANCH:"=%"
for /f "tokens=* delims= " %%I in ("%BACKUP_BRANCH%") do set "BACKUP_BRANCH=%%I"
:trim_backup_branch
if defined BACKUP_BRANCH if "%BACKUP_BRANCH:~-1%"==" " (
    set "BACKUP_BRANCH=%BACKUP_BRANCH:~0,-1%"
    goto :trim_backup_branch
)
set "BACKUP_BRANCH=%BACKUP_BRANCH: =_%"
if /I not "%BACKUP_BRANCH:~0,7%"=="backup/" set "BACKUP_BRANCH=backup/%BACKUP_BRANCH%"

git check-ref-format --branch "%BACKUP_BRANCH%" >nul 2>nul
if errorlevel 1 (
    echo.
    echo "%BACKUP_BRANCH%" is not a valid Git branch name.
    goto :end
)

git show-ref --verify --quiet "refs/heads/%BACKUP_BRANCH%"
if not errorlevel 1 (
    echo.
    echo Local branch %BACKUP_BRANCH% already exists.
    goto :end
)

git ls-remote --exit-code --heads %REMOTE% "%BACKUP_BRANCH%" >nul 2>nul
if not errorlevel 1 (
    echo.
    echo Remote branch %REMOTE%/%BACKUP_BRANCH% already exists.
    goto :end
)

echo.
echo Summary:
echo Backup branch : %BACKUP_BRANCH%
if /I "%CURRENT_BRANCH%"=="%TARGET_BRANCH%" (
    echo Current work  : will be committed on %TARGET_BRANCH% if needed, then pushed to %REMOTE%/%TARGET_BRANCH%
) else (
    echo Current work  : will be committed on %CURRENT_BRANCH% if needed
    echo Final push    : current HEAD from %CURRENT_BRANCH% will be pushed to %REMOTE%/%TARGET_BRANCH% if it is a safe fast-forward
)
echo.
set /p "CONFIRM=Type PUSH to create the backup and continue, or press Enter to abort: "
if /I not "%CONFIRM%"=="PUSH" (
    echo.
    echo Operation canceled. Nothing was changed.
    set "EXIT_CODE=0"
    goto :end
)

echo.
echo Creating local backup branch %BACKUP_BRANCH% from %REMOTE%/%TARGET_BRANCH%...
git branch --no-track "%BACKUP_BRANCH%" "refs/remotes/%REMOTE%/%TARGET_BRANCH%"
if errorlevel 1 (
    echo Failed to create local backup branch. Nothing was pushed.
    goto :end
)

echo.
echo Pushing backup branch %BACKUP_BRANCH% to %REMOTE%...
git push %REMOTE% "refs/heads/%BACKUP_BRANCH%:refs/heads/%BACKUP_BRANCH%"
if errorlevel 1 (
    echo Backup push failed.
    echo The local backup branch was created, but %TARGET_BRANCH% was not pushed.
    goto :end
)

echo.
echo Staging current changes...
git add -A
if errorlevel 1 (
    echo git add failed.
    echo The backup branch is already on %REMOTE%.
    goto :end
)

git diff --cached --quiet >nul 2>nul
if errorlevel 1 (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "STAMP=%%I"
    echo Creating commit: Sync: %STAMP%
    git commit -m "Sync: %STAMP%"
    if errorlevel 1 (
        echo Commit failed.
        echo The backup branch is already on %REMOTE%.
        goto :end
    )
) else (
    echo No local changes to commit.
)

if /I "%CURRENT_BRANCH%"=="%TARGET_BRANCH%" (
    echo.
    echo Rebasing latest %REMOTE%/%TARGET_BRANCH% onto local %TARGET_BRANCH%...
    git pull --rebase %REMOTE% %TARGET_BRANCH%
    if errorlevel 1 (
        echo Rebase failed.
        echo Resolve the issue, then rerun when ready.
        echo The backup branch is already on %REMOTE%.
        goto :end
    )

    echo.
    echo Pushing %TARGET_BRANCH% to %REMOTE%...
    git push -u %REMOTE% %TARGET_BRANCH%
    if errorlevel 1 (
        echo Push failed.
        echo The backup branch is already on %REMOTE%.
        goto :end
    )
) else (
    echo.
    echo Checking whether %CURRENT_BRANCH% safely contains the latest %REMOTE%/%TARGET_BRANCH%...
    git merge-base --is-ancestor "refs/remotes/%REMOTE%/%TARGET_BRANCH%" HEAD >nul 2>nul
    if errorlevel 1 (
        echo Current branch does not yet contain the latest %REMOTE%/%TARGET_BRANCH%.
        echo The backup branch is already on %REMOTE%.
        echo Merge or rebase %REMOTE%/%TARGET_BRANCH% into %CURRENT_BRANCH%, then run this script again.
        goto :end
    )

    echo.
    echo Pushing current HEAD from %CURRENT_BRANCH% to %REMOTE%/%TARGET_BRANCH%...
    git push %REMOTE% "HEAD:refs/heads/%TARGET_BRANCH%"
    if errorlevel 1 (
        echo Push to %REMOTE%/%TARGET_BRANCH% failed.
        echo The backup branch is already on %REMOTE%.
        goto :end
    )
)

echo.
echo Backup and push completed successfully.
echo Backup branch: %BACKUP_BRANCH%
echo Your current checkout stayed on %CURRENT_BRANCH%.
set "EXIT_CODE=0"

:end
echo.
pause
exit /b %EXIT_CODE%
