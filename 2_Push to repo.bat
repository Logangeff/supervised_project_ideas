@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "REPO_URL=https://github.com/Logangeff/supervised_project_ideas.git"
set "REMOTE=origin"
set "DEFAULT_BRANCH=main"
set "MENU_FILE=%TEMP%\supervised_project_push_menu_%RANDOM%_%RANDOM%.txt"

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

if "%BRANCH_COUNT%"=="0" (
    set /a BRANCH_COUNT=1
    >>"%MENU_FILE%" echo(1^|__MAIN_EMPTY__^|
)

set /a NEW_BRANCH_OPTION=BRANCH_COUNT+1
>>"%MENU_FILE%" echo(%NEW_BRANCH_OPTION%^|__NEW__^|

echo.
echo ============================================================
echo Push To Repo
echo ============================================================
echo Choose where to push the current work:
echo.
for /f "usebackq tokens=1-3 delims=|" %%A in ("%MENU_FILE%") do (
    if /I "%%B"=="__MAIN_EMPTY__" (
        echo   %%A. Push to %DEFAULT_BRANCH% ^(Recommended for empty repo^)
    ) else if /I "%%B"=="__NEW__" (
        echo   %%A. Create a new branch...
    ) else (
        echo   %%A. %%B ^(%%C^)
    )
)
echo.

set "BRANCH_CHOICE=1"
set /p "BRANCH_CHOICE=Enter branch number to push to [1]: "
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
set "TARGET_LABEL="
set "TARGET_IS_NEW=0"

for /f "usebackq tokens=1-3 delims=|" %%A in ("%MENU_FILE%") do (
    if "%%A"=="!BRANCH_CHOICE_NUM!" (
        set "TARGET_BRANCH=%%B"
        set "TARGET_LABEL=%%B"
    )
)
if not defined TARGET_BRANCH (
    echo.
    echo Invalid branch selection.
    pause
    exit /b 1
)
if /I "%TARGET_BRANCH%"=="__MAIN_EMPTY__" (
    set "TARGET_BRANCH=%DEFAULT_BRANCH%"
    set "TARGET_LABEL=%DEFAULT_BRANCH% (recommended)"
    set "TARGET_IS_NEW=1"
    goto branch_choice_done
)
if /I "%TARGET_BRANCH%"=="__NEW__" goto do_new_branch
goto branch_choice_done

:do_new_branch
call :prompt_new_branch
set "PROMPT_EXIT=%ERRORLEVEL%"
if "%PROMPT_EXIT%"=="2" (
    pause
    exit /b 0
)
if not "%PROMPT_EXIT%"=="0" (
    pause
    exit /b 1
)
goto branch_choice_done

:prompt_new_branch
setlocal DisableDelayedExpansion
echo.
echo Type a new branch name or a normal sentence.
echo Spaces will be converted to underscores.
set "RAW_BRANCH="
set /p "RAW_BRANCH=New branch name or phrase (press Enter to abort): "
if not defined RAW_BRANCH (
    echo.
    echo Push canceled.
    endlocal & exit /b 2
)

set "TARGET_BRANCH=%RAW_BRANCH:"=%"
for /f "tokens=* delims= " %%I in ("%TARGET_BRANCH%") do set "TARGET_BRANCH=%%I"
:trim_target_branch
if defined TARGET_BRANCH if "%TARGET_BRANCH:~-1%"==" " (
    set "TARGET_BRANCH=%TARGET_BRANCH:~0,-1%"
    goto :trim_target_branch
)
set "TARGET_BRANCH=%TARGET_BRANCH: =_%"
if not defined TARGET_BRANCH (
    echo.
    echo Push canceled.
    endlocal & exit /b 2
)

git check-ref-format --branch "%TARGET_BRANCH%" >nul 2>nul
if errorlevel 1 goto prompt_new_branch_invalid

git ls-remote --exit-code --heads %REMOTE% "%TARGET_BRANCH%" >nul 2>nul
if not errorlevel 1 goto prompt_new_branch_exists

endlocal & set "TARGET_BRANCH=%TARGET_BRANCH%" & set "TARGET_IS_NEW=1" & set "TARGET_LABEL=%TARGET_BRANCH% (new)"
exit /b 0

:prompt_new_branch_invalid
echo.
echo "%TARGET_BRANCH%" is not a valid Git branch name.
endlocal & exit /b 1

:prompt_new_branch_exists
echo.
echo %REMOTE%/%TARGET_BRANCH% already exists.
echo Choose it from the numbered list instead.
endlocal & exit /b 1

:branch_choice_done
set "CURRENT_BRANCH="
for /f "usebackq delims=" %%I in (`git symbolic-ref --short -q HEAD`) do set "CURRENT_BRANCH=%%I"
if not defined CURRENT_BRANCH set "CURRENT_BRANCH=(detached HEAD)"

echo.
echo Current branch : %CURRENT_BRANCH%
echo Push target    : %TARGET_LABEL%
echo.
echo Local changes that would be pushed from this PC:
git status --short
echo.
set /p "CONFIRM=Type PUSH to commit and push to %TARGET_BRANCH%, or press Enter to abort: "
if /I not "%CONFIRM%"=="PUSH" (
    echo.
    echo Push canceled.
    pause
    exit /b 0
)

git add -A
git diff --cached --quiet >nul 2>nul
if errorlevel 1 (
    for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"`) do set "STAMP=%%I"
    git commit -m "Sync: %STAMP%"
    if errorlevel 1 (
        echo.
        echo Commit failed.
        pause
        exit /b 1
    )
) else (
    echo No local changes to commit.
)

git fetch %REMOTE% "%TARGET_BRANCH%" >nul 2>nul
set "REMOTE_BRANCH_EXISTS=0"
if not errorlevel 1 set "REMOTE_BRANCH_EXISTS=1"

if "%TARGET_IS_NEW%"=="1" goto push_new_branch
if "%REMOTE_BRANCH_EXISTS%"=="0" goto push_new_branch

if /I "%CURRENT_BRANCH%"=="%TARGET_BRANCH%" (
    echo.
    echo Rebasing latest %REMOTE%/%TARGET_BRANCH% onto local %TARGET_BRANCH%...
    git pull --rebase %REMOTE% "%TARGET_BRANCH%"
    if errorlevel 1 (
        echo.
        echo Pull --rebase failed. Resolve the issue, then run this script again.
        pause
        exit /b 1
    )

    echo.
    echo Pushing %TARGET_BRANCH% to %REMOTE%...
    git push -u %REMOTE% "%TARGET_BRANCH%"
    if errorlevel 1 (
        echo.
        echo Push failed.
        pause
        exit /b 1
    )
    goto push_done
)

echo.
echo Checking whether current HEAD already contains the latest %REMOTE%/%TARGET_BRANCH%...
git merge-base --is-ancestor "refs/remotes/%REMOTE%/%TARGET_BRANCH%" HEAD >nul 2>nul
if errorlevel 1 (
    echo.
    echo Current HEAD does not contain the latest %REMOTE%/%TARGET_BRANCH%.
    echo Rebase or merge that branch first, then run this script again.
    pause
    exit /b 1
)

echo.
echo Pushing current HEAD to %REMOTE%/%TARGET_BRANCH%...
git push %REMOTE% "HEAD:refs/heads/%TARGET_BRANCH%"
if errorlevel 1 (
    echo.
    echo Push failed.
    pause
    exit /b 1
)
goto push_done

:push_new_branch
echo.
echo Creating and pushing new branch %TARGET_BRANCH%...
git push -u %REMOTE% "HEAD:refs/heads/%TARGET_BRANCH%"
if errorlevel 1 (
    echo.
    echo Push failed.
    pause
    exit /b 1
)

:push_done
echo.
echo Project is now pushed to %TARGET_BRANCH%.
if exist "%MENU_FILE%" del "%MENU_FILE%" >nul 2>nul
pause
exit /b 0
