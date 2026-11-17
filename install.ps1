<#
  TAPEDECK - Windows installer (runs the Linux install inside WSL2).

  From PowerShell:
    irm https://raw.githubusercontent.com/CharlesMod/infinite-tapedeck/main/install.ps1 -OutFile install.ps1
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Music "$env:USERPROFILE\Music"

  Why WSL2: TAPEDECK coordinates the GPU with Linux process tools (/proc,
  POSIX signals, systemd). WSL2 is a real Linux with CUDA passthrough to your
  NVIDIA card, so the tested Linux installer runs unchanged inside it.

  What it does (re-running is safe; finished steps are skipped):
    1. checks the NVIDIA driver and VRAM on Windows
    2. installs WSL2 if missing (needs admin + one reboot, then run it again)
    3. installs Ubuntu if missing - you create a Linux username once
    4. checks the GPU is visible inside WSL, installs git/curl/ffmpeg
    5. copies your music into the Linux filesystem (audio files only; reading
       it over /mnt/c would make every capture far slower) - or links it
       with -LinkMusic
    6. runs install.sh inside WSL: ComfyUI, CUDA PyTorch, weights, the deck
    7. adds "TAPEDECK" and "TAPEDECK - stop" to the Start menu and desktop,
       and opens the deck

  Options:
    -Music DIR       your music folder (Windows path)
    -LinkMusic       read the music in place from /mnt/c instead of copying
    -Distro NAME     WSL distribution (default Ubuntu-24.04)
    -Port N          deck port (default 8188)
    -NoStart         install only
#>
param(
  [string]$Music = "",
  [switch]$LinkMusic,
  [string]$Distro = "Ubuntu-24.04",
  [int]$Port = 8188,
  [switch]$NoStart
)
$ErrorActionPreference = "Stop"
$env:WSL_UTF8 = "1"   # newer wsl.exe prints UTF-8 instead of UTF-16 when set

function Step($m) { Write-Host ""; Write-Host "> $m" -ForegroundColor Green }
function Ok($m)   { Write-Host "  ok  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "  !   $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host ""; Write-Host "x $m" -ForegroundColor Red; exit 1 }
function Clean($s) { ($s | Out-String) -replace "`0", "" }   # strip UTF-16 NULs from old wsl.exe

function Test-Admin {
  $p = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
  return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Run a bash script inside the distro. The script is written to a file with
# LF line endings and run by path, so no PowerShell -> bash quoting is involved.
function Invoke-Wsl([string]$Script, [switch]$AsRoot) {
  $tmp = Join-Path $env:TEMP ("tapedeck-" + [guid]::NewGuid().ToString("N") + ".sh")
  [IO.File]::WriteAllText($tmp, ($Script -replace "`r`n", "`n"), (New-Object Text.UTF8Encoding($false)))
  $linux = Clean (wsl.exe -d $Distro -- wslpath -a ($tmp -replace '\\', '/'))
  $linux = $linux.Trim()
  # Out-Host: show the output, and keep it out of this function's return
  # value (PowerShell returns everything a function emits, not just `return`)
  if ($AsRoot) { wsl.exe -d $Distro -u root -- bash $linux | Out-Host }
  else { wsl.exe -d $Distro -- bash -l $linux | Out-Host }
  $code = $LASTEXITCODE
  Remove-Item $tmp -ErrorAction SilentlyContinue
  return $code
}

# ------------------------------------------------------------------ 1. GPU
Step "checking the NVIDIA GPU"
$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) { Die "nvidia-smi not found: install the current NVIDIA Game Ready / Studio driver first" }
$gpu = (& nvidia-smi.exe --query-gpu=name,memory.total --format=csv,noheader,nounits | Select-Object -First 1).Split(",")
$gpuName = $gpu[0].Trim(); $gpuMb = [int]$gpu[1].Trim()
Ok "$gpuName ($gpuMb MiB)"
if ($gpuMb -lt 15000) { Warn "MiniMax Music 3 wants a 16 GB card; $gpuMb MiB will likely run out of memory" }

# ------------------------------------------------------------------ 2. WSL
Step "WSL2"
wsl.exe --status *> $null
if ($LASTEXITCODE -ne 0) {
  if (-not (Test-Admin)) {
    Warn "WSL is not installed; asking for administrator rights to install it"
    $argList = "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
    foreach ($k in $PSBoundParameters.Keys) {
      $v = $PSBoundParameters[$k]
      if ($v -is [switch]) { if ($v) { $argList += " -$k" } } else { $argList += " -$k `"$v`"" }
    }
    Start-Process powershell -Verb RunAs -ArgumentList $argList -Wait
    exit 0
  }
  wsl.exe --install --no-distribution
  Write-Host ""
  Write-Host "WSL2 is installed. REBOOT, then run this same command again to continue." -ForegroundColor Yellow
  exit 0
}
wsl.exe --set-default-version 2 *> $null
Ok "WSL2 available"

# ------------------------------------------------------------------ 3. distro
Step "Linux distribution ($Distro)"
$have = (Clean (wsl.exe -l -q)) -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
if ($have -notcontains $Distro) {
  Write-Host "  installing $Distro. When it asks, create a Linux username and password"
  Write-Host "  (they are only for this Linux system), then type 'exit' to come back here."
  wsl.exe --install -d $Distro
  if ($LASTEXITCODE -ne 0) { Die "could not install $Distro (try: wsl --install -d $Distro)" }
  $have = (Clean (wsl.exe -l -q)) -split "`r?`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }
  if ($have -notcontains $Distro) {
    Read-Host "  finish the Ubuntu setup window (create the user), then press Enter here"
  }
}
$user = (Clean (wsl.exe -d $Distro -- whoami)).Trim()
if (-not $user) { Die "$Distro is not set up yet: run 'wsl -d $Distro' once, create the user, then re-run" }
if ($user -eq "root") { Warn "the default WSL user is root; it works, but a normal user is recommended" }
Ok "$Distro, user $user"

# ------------------------------------------------------------------ 4. inside WSL
Step "GPU and system packages inside WSL"
$code = Invoke-Wsl -Script @'
set -e
if ! nvidia-smi >/dev/null 2>&1 && ! /usr/lib/wsl/lib/nvidia-smi >/dev/null 2>&1; then
  echo "GPU not visible inside WSL: update the NVIDIA Windows driver, then run 'wsl --update'"
  exit 3
fi
'@
if ($code -eq 3) { Die "the GPU is not visible inside WSL (update the NVIDIA driver, run 'wsl --update')" }
Ok "GPU visible inside WSL"
$code = Invoke-Wsl -AsRoot -Script @'
set -e
export DEBIAN_FRONTEND=noninteractive
need=""
for p in git curl ffmpeg rsync ca-certificates; do dpkg -s $p >/dev/null 2>&1 || need="$need $p"; done
if [ -n "$need" ]; then apt-get update -qq && apt-get install -y -qq $need >/dev/null; fi
echo packages ok
'@
if ($code -ne 0) { Die "could not install git/curl/ffmpeg inside $Distro" }
Ok "git, curl, ffmpeg, rsync"

# ------------------------------------------------------------------ 5. music
$wslMusic = ""
if ($Music) {
  Step "music"
  if (-not (Test-Path -LiteralPath $Music -PathType Container)) { Die "-Music: not a folder: $Music" }
  if ($Music.Contains("'")) { Die "-Music: please use a folder path without an apostrophe" }
  $src = (Clean (wsl.exe -d $Distro -- wslpath -a ((Resolve-Path -LiteralPath $Music).Path -replace '\\', '/'))).Trim()
  if ($LinkMusic) {
    $wslMusic = $src
    Warn "reading music in place from $src (slower captures; copy instead for speed)"
  } else {
    Write-Host "  copying audio files into Linux (only the first run copies everything)"
    $code = Invoke-Wsl -Script @"
set -e
mkdir -p ~/tapedeck-music
rsync -a --prune-empty-dirs --include='*/' \
  --include='*.flac' --include='*.FLAC' --include='*.mp3' --include='*.MP3' \
  --include='*.ogg' --include='*.m4a' --include='*.opus' --include='*.wav' \
  --include='*.aac' --include='*.aiff' --include='*.wma' --exclude='*' \
  '$src/' ~/tapedeck-music/
echo "`$(find ~/tapedeck-music -type f | wc -l) audio files in ~/tapedeck-music"
"@
    if ($code -ne 0) { Die "copying the music failed" }
    $wslMusic = "~/tapedeck-music"
  }
  Ok "music ready"
}

# ------------------------------------------------------------------ 6. install.sh
Step "installing TAPEDECK inside WSL (the long part: ~8 GB environment + ~14 GB weights)"
$musicArg = ""
if ($wslMusic) { $musicArg = "--music $wslMusic" }
if ($wslMusic -and $wslMusic -notlike "~*") { $musicArg = "--music '$wslMusic'" }
$startArg = "--no-start"   # Windows starts it through the shortcut below, which keeps WSL awake
$code = Invoke-Wsl -Script @"
set -e
if [ -d ~/tapedeck/.git ]; then git -C ~/tapedeck pull -q --ff-only || true
else git clone -q https://github.com/CharlesMod/infinite-tapedeck ~/tapedeck; fi
~/tapedeck/install.sh $musicArg --port $Port $startArg -y
"@
if ($code -ne 0) { Die "install.sh failed inside WSL (see above); fix and run this again - finished steps are skipped" }
Ok "installed in WSL at ~/tapedeck"

# ------------------------------------------------------------------ 7. shortcuts
Step "Start menu and desktop shortcuts"
$home_ = Join-Path $env:LOCALAPPDATA "Tapedeck"
New-Item -ItemType Directory -Force -Path $home_ | Out-Null
$url = "http://localhost:$Port/extensions/music_studio/index.html"
# WSL stops a distro soon after its last console closes, which would stop the
# radio too - so the launcher stays open (minimised) while the deck runs.
$startCmd = @"
@echo off
title TAPEDECK - keep this window open (minimise it); close it to stop
wsl.exe -d $Distro -- bash -lc "~/.local/bin/tapedeck start"
start "" "$url"
echo.
echo  TAPEDECK is running: $url
echo  Keep this window open. Close it (or use "TAPEDECK - stop") to stop.
echo.
wsl.exe -d $Distro -- bash -lc "~/.local/bin/tapedeck logs"
"@
$stopCmd = @"
@echo off
wsl.exe -d $Distro -- bash -lc "~/.local/bin/tapedeck stop"
"@
[IO.File]::WriteAllText((Join-Path $home_ "tapedeck-start.cmd"), ($startCmd -replace "(?<!`r)`n", "`r`n"))
[IO.File]::WriteAllText((Join-Path $home_ "tapedeck-stop.cmd"), ($stopCmd -replace "(?<!`r)`n", "`r`n"))

$shell = New-Object -ComObject WScript.Shell
$menu = Join-Path ([Environment]::GetFolderPath("Programs")) "TAPEDECK"
New-Item -ItemType Directory -Force -Path $menu | Out-Null
function New-Shortcut($path, $target, $min) {
  $s = $shell.CreateShortcut($path)
  $s.TargetPath = $target
  $s.WorkingDirectory = $home_
  if ($min) { $s.WindowStyle = 7 }
  $s.Save()
}
New-Shortcut (Join-Path $menu "TAPEDECK.lnk") (Join-Path $home_ "tapedeck-start.cmd") $true
New-Shortcut (Join-Path $menu "TAPEDECK - stop.lnk") (Join-Path $home_ "tapedeck-stop.cmd") $false
New-Shortcut (Join-Path ([Environment]::GetFolderPath("Desktop")) "TAPEDECK.lnk") (Join-Path $home_ "tapedeck-start.cmd") $true
Ok "Start menu > TAPEDECK, and a desktop shortcut"

Write-Host ""
Write-Host "TAPEDECK is installed." -ForegroundColor Green
Write-Host "  Open it from the TAPEDECK shortcut (keeps a small window open while it runs)."
Write-Host "  Deck: $url"
Write-Host "  Press 'Accept & start listening pass' in the deck to allow the one-time"
Write-Host "  17 GB NVIDIA Music Flamingo download - recommended."
if (-not $NoStart) {
  Start-Process -FilePath (Join-Path $home_ "tapedeck-start.cmd") -WindowStyle Minimized
  # the first capture starts once the deck answers (ComfyUI can take a minute)
  Write-Host "  waiting for the deck to come up, then starting the first capture..."
  $code = Invoke-Wsl -Script @'
for i in $(seq 90); do
  if ~/.local/bin/tapedeck capture 2>/dev/null | grep -qiE "ok|already running"; then echo "first capture started"; exit 0; fi
  sleep 4
done
echo "the deck did not answer in 6 minutes - start the capture from the deck or with: tapedeck capture"
'@
}
