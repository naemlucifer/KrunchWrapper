@echo off
setlocal enabledelayedexpansion

:: KrunchWrapper Cleanup Script (Windows)
:: Removes all contents from logs\ and temp\ folders

:: Get the directory of the script
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

echo [INFO] KrunchWrapper Cleanup Script
echo [INFO] Working directory: %CD%
echo ================================================

:: Function to safely remove folder contents
goto :main

:cleanup_folder
set "folder=%~1"
if exist "%folder%" (
    echo [INFO] Cleaning %folder%...
    
    :: Count files in the folder
    set file_count=0
    for /f %%i in ('dir /b /s /a-d "%folder%" 2^>NUL ^| find /c /v ""') do set file_count=%%i
    
    if !file_count! GTR 0 (
        echo [INFO]    Found !file_count! files to remove
        rmdir /s /q "%folder%" >NUL 2>&1
        if !ERRORLEVEL! EQU 0 (
            mkdir "%folder%" >NUL 2>&1
            echo [SUCCESS]    Cleaned %folder% ^(removed !file_count! files^)
        ) else (
            echo [ERROR]    Failed to clean %folder%
            set cleanup_errors=1
        )
    ) else (
        echo [INFO]    %folder% is already empty
    )
) else (
    echo [WARNING]    %folder% does not exist, creating it...
    mkdir "%folder%" >NUL 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [SUCCESS]    Created %folder%
    ) else (
        echo [ERROR]    Failed to create %folder%
        set cleanup_errors=1
    )
)
goto :eof

:main
set cleanup_errors=0

:: Check if we're in the right directory
if not exist "README.md" (
    echo [ERROR] This doesn't appear to be the KrunchWrapper root directory
    echo [ERROR] Expected to find README.md and core\ folder
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

if not exist "core" (
    echo [ERROR] This doesn't appear to be the KrunchWrapper root directory
    echo [ERROR] Expected to find README.md and core\ folder
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

echo [SUCCESS] Verifying KrunchWrapper project structure...
echo [SUCCESS]    Found README.md
echo [SUCCESS]    Found core\ directory

:: Clean up logs folder
call :cleanup_folder "logs"

:: Clean up temp folder  
call :cleanup_folder "temp"

:: Also clean any .log files in root directory
echo [INFO] Cleaning .log files in root directory...
set log_count=0
for %%f in (*.log) do (
    if exist "%%f" set /a log_count+=1
)

if !log_count! GTR 0 (
    echo [INFO]    Found !log_count! .log files to remove
    del /q *.log >NUL 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [SUCCESS]    Removed !log_count! .log files from root
    ) else (
        echo [ERROR]    Failed to remove some .log files
        set cleanup_errors=1
    )
) else (
    echo [INFO]    No .log files found in root directory
)

:: Clean Python cache files
echo [INFO] Cleaning Python cache files...
set cache_count=0

:: Count __pycache__ directories
for /f %%i in ('dir /b /s /ad "__pycache__" 2^>NUL ^| find /c /v ""') do set cache_count=%%i

if !cache_count! GTR 0 (
    echo [INFO]    Found !cache_count! __pycache__ directories to remove
    
    :: Remove all __pycache__ directories
    for /f "delims=" %%d in ('dir /b /s /ad "__pycache__" 2^>NUL') do (
        rmdir /s /q "%%d" >NUL 2>&1
    )
    echo [SUCCESS]    Removed Python cache directories
) else (
    echo [INFO]    No __pycache__ directories found
)

:: Clean pytest cache
echo [INFO] Cleaning pytest cache...
set pytest_cache_count=0

if exist ".pytest_cache" (
    set /a pytest_cache_count+=1
)
if exist ".cache" (
    set /a pytest_cache_count+=1
)

if !pytest_cache_count! GTR 0 (
    echo [INFO]    Found !pytest_cache_count! pytest cache directories to remove
    
    if exist ".pytest_cache" (
        rmdir /s /q ".pytest_cache" >NUL 2>&1
        if !ERRORLEVEL! EQU 0 (
            echo [SUCCESS]    Removed .pytest_cache\
        ) else (
            echo [ERROR]    Failed to remove .pytest_cache\
            set cleanup_errors=1
        )
    )
    
    if exist ".cache" (
        rmdir /s /q ".cache" >NUL 2>&1
        if !ERRORLEVEL! EQU 0 (
            echo [SUCCESS]    Removed .cache\
        ) else (
            echo [ERROR]    Failed to remove .cache\
            set cleanup_errors=1
        )
    )
) else (
    echo [INFO]    No pytest cache directories found
)

:: Optional: Clean additional Windows-specific temp files
echo [INFO] Cleaning Windows-specific temporary files...
set temp_files_count=0

:: Count .tmp files
for %%f in (*.tmp) do (
    if exist "%%f" set /a temp_files_count+=1
)

:: Count .bak files  
for %%f in (*.bak) do (
    if exist "%%f" set /a temp_files_count+=1
)

if !temp_files_count! GTR 0 (
    echo [INFO]    Found !temp_files_count! temporary files to remove
    del /q *.tmp >NUL 2>&1
    del /q *.bak >NUL 2>&1
    echo [SUCCESS]    Removed temporary files
) else (
    echo [INFO]    No additional temporary files found
)

:: Final summary
echo.
if !cleanup_errors! EQU 0 (
    echo [SUCCESS] Cleanup completed successfully!
) else (
    echo [WARNING] Cleanup completed with some errors!
    echo [INFO] Some files may be in use and couldn't be removed
)

echo [INFO] Summary:
echo [INFO]    • Cleaned logs\ folder
echo [INFO]    • Cleaned temp\ folder
echo [INFO]    • Removed .log files from root
echo [INFO]    • Removed Python cache files
echo [INFO]    • Removed pytest cache directories
echo [INFO]    • Removed Windows temporary files
echo.
echo [INFO] Tip: Run this script anytime to clean up KrunchWrapper generated files
echo.
echo Press any key to exit...
pause >nul 