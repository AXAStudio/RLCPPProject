# --- ENVIRONMENT FIX ---
# This forces the script to know where the compiler is, regardless of Windows settings
$MSYS_PATH = "C:\msys64\ucrt64\bin"
$USR_PATH = "C:\msys64\usr\bin"

if ($env:PATH -notlike "*$MSYS_PATH*") {
    $env:PATH = "$MSYS_PATH;$USR_PATH;" + $env:PATH
}

function build-game {
    Write-Host "--- Compiling main.cpp with Raylib ---" -ForegroundColor Cyan

    Push-Location $PSScriptRoot
    try {
        # The exact folder where we found raylib.h
        $RAYLIB_SRC = "C:/raylib/raylib/src"

        & "C:/msys64/ucrt64/bin/g++.exe" main.cpp -o mygame.exe `
            -O2 -DNDEBUG -std=gnu++17 `
            "-I$RAYLIB_SRC" `
            "-L$RAYLIB_SRC" `
            -lraylib -lopengl32 -lgdi32 -lwinmm 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
            & .\mygame.exe
        } else {
            Write-Host "BUILD FAILED." -ForegroundColor Red
        }
    } finally {
        Pop-Location
    }
}

function run-game {
    Push-Location $PSScriptRoot
    try {
        if (Test-Path .\mygame.exe) {
            & .\mygame.exe
        } else {
            Write-Host "Error: mygame.exe not found. Run build-game first." -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
}

Write-Host "Environment loaded. You can now use 'build-game' or 'run-game'." -ForegroundColor Magenta
