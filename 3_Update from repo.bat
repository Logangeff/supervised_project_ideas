@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "REPO_URL=https://github.com/Logangeff/supervised_project_ideas.git"
set "REMOTE=origin"
set "DEFAULT_BRANCH=main"
set "MENU_FILE=%TEMP%\supervised_project_update_menu_%RANDOM%_%RANDOM%.txt"

git --version >nul 2>nul
if errorlevel 1 (
    echo Git is not installed or not available on PATH.
    pause
    exit /b 1
)

git rev-parse --git-dir >nul 2>nul
if errorlevel 1 (
    echo Initializing local git repository...
    git init
    if errorlevel 1 (
        pause
        exit /b 1
    )
)

git remote get-url %REMOTE% >nul 2>nul
if errorlevel 1 (
    git remote add %REMOTE% "%REPO_URL%"
) else (
    git remote set-url %REMOTE% "%REPO_URL%"
)

echo.
echo Fetching remote branches...
git fetch %REMOTE% --prune
if errorlevel 1 (
    echo.
    echo Fetch failed.
    pause
    exit /b 1
)

if exist "%MENU_FILE%" del "%MENU_FILE%" >nul 2>nul
set "BRANCH_COUNT=0"
for /f "usebackq tokens=1* delims=|" %%A in (`git for-each-ref "refs/remotes/%REMOTE%/%DEFAULT_BRANCH%" --format="%%(refname:strip=3)|%%(committerdate:iso8601)"`) do (
    if /I not "%%A"=="HEAD" (
        set /a BRANCH_COUNT+=1
        >>"%MENU_FILE%" echo(!BRANCH_COUNT!^|%%A^|%%B
    )
)
for /f "usebackq tokens=1* delims=|" %%A in (`git for-each-ref "refs/remotes/%REMOTE%" --sort=-committerdate --format="%%(refname:strip=3)|%%(committerdate:iso8601)"`) do (
    if /I not "%%A"=="HEAD" if /I not "%%A"=="%DEFAULT_BRANCH%" (
        set /a BRANCH_COUNT+=1
        >>"%MENU_FILE%" echo(!BRANCH_COUNT!^|%%A^|%%B
    )
)

if %BRANCH_COUNT% LEQ 0 (
    echo.
    echo No remote branches were found on %REMOTE%.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Update From Repo
echo ============================================================
echo Choose which remote branch to update from:
echo.
for /f "usebackq tokens=1-3 delims=|" %%A in ("%MENU_FILE%") do (
    echo   %%A. %%B ^(%%C^)
)
echo.

set "BRANCH_CHOICE=1"
set /p "BRANCH_CHOICE=Enter branch number to update from [1]: "
if not defined BRANCH_CHOICE set "BRANCH_CHOICE=1"

echo(%BRANCH_CHOICE%| findstr /r "^[1-9][0-9]*$" >nul
if errorlevel 1 (
    echo.
    echo Invalid branch selection.
    pause
    exit /b 1
)
set /a BRANCH_CHOICE_NUM=BRANCH_CHOICE

set "TARGET_BRANCH="
for /f "usebackq tokens=1-3 delims=|" %%A in ("%MENU_FILE%") do (
    if "%%A"=="!BRANCH_CHOICE_NUM!" set "TARGET_BRANCH=%%B"
)
if not defined TARGET_BRANCH (
    echo.
    echo Invalid branch selection.
    pause
    exit /b 1
)

echo.
echo WARNING: this script can overwrite local Git changes in this folder.
echo Local changes that may be stashed or replaced by %REMOTE%/%TARGET_BRANCH%:
git status --short
echo.
set /p "CONFIRM=Type Y to update this PC from %TARGET_BRANCH%, or press Enter to abort: "
if /I "%CONFIRM%"=="Y" goto continue_update
if /I "%CONFIRM%"=="YES" goto continue_update
echo.
echo Update canceled.
pause
exit /b 0

:continue_update
git diff --quiet >nul 2>nul
if errorlevel 1 goto stash_changes
git diff --cached --quiet >nul 2>nul
if errorlevel 1 goto stash_changes
set "HAS_UNTRACKED="
for /f "usebackq delims=" %%I in (`git ls-files --others --exclude-standard`) do set "HAS_UNTRACKED=1"
if defined HAS_UNTRACKED goto stash_changes
goto checkout_branch

:stash_changes
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "STAMP=%%I"
git stash push --include-untracked -m "Auto-stash before sync: %STAMP%"

:checkout_branch
git checkout "%TARGET_BRANCH%" >nul 2>nul
if errorlevel 1 (
    git checkout -B "%TARGET_BRANCH%" "%REMOTE%/%TARGET_BRANCH%"
)

git reset --hard "%REMOTE%/%TARGET_BRANCH%"
if errorlevel 1 (
    echo.
    echo Reset failed.
    pause
    exit /b 1
)

git clean -fd
if errorlevel 1 (
    echo.
    echo Clean failed.
    pause
    exit /b 1
)

echo.
echo Local project now matches %REMOTE%/%TARGET_BRANCH%.
if exist "%MENU_FILE%" del "%MENU_FILE%" >nul 2>nul
pause
exit /b 0
