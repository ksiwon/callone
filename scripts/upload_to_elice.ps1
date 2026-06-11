# callone 업로드 한방 스크립트 (로컬 Windows → 엘리스 H100)
# 폴더에서 불필요한 것(.venv/node_modules/로컬결과물) 빼고 tar 로 묶어 scp 업로드.
#
# 사용법 (PowerShell):
#   .\scripts\upload_to_elice.ps1 -Pem "$env:USERPROFILE\.ssh\elice-xxxx.pem" -Port 12345
#   (필요시 -ServerUser / -ServerHost 지정. 엘리스 연결정보에서 확인)
#
# 끝나면 서버에서:  tar -xf ~/callone.tar && cd callone && bash scripts/setup_server.sh pilot

param(
  [Parameter(Mandatory = $true)][string]$Pem,      # .pem 개인키 경로
  [Parameter(Mandatory = $true)][int]$Port,        # 엘리스 SSH 포트
  [string]$ServerUser = "elicer",
  [string]$ServerHost = "central-01.tcp.tunnel.elice.io"
)
$ErrorActionPreference = "Stop"

$callone = Split-Path -Parent $PSScriptRoot         # ...\callone
$parent  = Split-Path -Parent $callone              # ...\coding
$name    = Split-Path -Leaf $callone                # callone
$tar     = Join-Path $parent "callone.tar"

if (-not (Test-Path $Pem)) { throw "키 파일 없음: $Pem" }

Write-Host "[1/4] 키 권한 설정..." -ForegroundColor Cyan
icacls $Pem /inheritance:r | Out-Null
icacls $Pem /grant:r "$($env:USERNAME):(R)" | Out-Null

Write-Host "[2/4] 폴더 묶는 중 (불필요한 것 제외)..." -ForegroundColor Cyan
Push-Location $parent
$excludes = @(
  ".venv", ".git", ".hf_cache", "ui/node_modules",
  "data/wav16k", "data/restored", "data/diarized",
  "data/datasets", "data/speakers", "db", "models"
) | ForEach-Object { "--exclude=$name/$_" }
& tar -cf $tar @excludes $name
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "tar 실패" }
Pop-Location
$sizeGB = [math]::Round((Get-Item $tar).Length / 1GB, 2)
Write-Host "    → $tar ($sizeGB GB)" -ForegroundColor Green

Write-Host "[3/4] 업로드 (scp)... 시간 좀 걸림" -ForegroundColor Cyan
# StrictHostKeyChecking=accept-new: 첫 접속 호스트키 프롬프트 자동 수락(스크립트 안에서 입력 막힘 방지)
& scp -o StrictHostKeyChecking=accept-new -i $Pem -P $Port $tar "${ServerUser}@${ServerHost}:~/"
if ($LASTEXITCODE -ne 0) { throw "scp 실패 — 포트/키/호스트 확인" }

Write-Host "[4/4] 완료!" -ForegroundColor Green
Write-Host ""
Write-Host "이제 서버 접속해서 한 줄 실행:" -ForegroundColor Yellow
Write-Host "  ssh -i `"$Pem`" ${ServerUser}@${ServerHost} -p $Port" -ForegroundColor Yellow
Write-Host "  tar -xf ~/callone.tar && cd callone && bash scripts/setup_server.sh pilot" -ForegroundColor Yellow
