<#
sweep.ps1 - Phase 2 voice-LoRA sweep orchestrator (PowerShell drives WSL).

Trains the same dataset across bases x hot configs, then gates each adapter through
the VOICE eval (eval_compare.py) + mechanical score (score_run.py). Facts are NOT
graded here -- they come from retrieval (eval_system.py). Run from the training dir.

Orchestrate from PowerShell, NOT Git Bash (which mangles \\wsl$ and /mnt/c paths).

Usage:
  pwsh -File sweep.ps1                       # default 2 bases x 2 configs
  pwsh -File sweep.ps1 -Data data/train.jsonl
#>
param(
  [string]$Data = "data/train.jsonl",
  [string[]]$Bases = @("unsloth/Qwen3-4B", "unsloth/Llama-3.2-3B-Instruct"),
  [int[]]$Epochs = @(6, 8),
  [double]$Lr = 2e-4,
  [int]$MaxNew = 200,
  [string]$WslDir = "\\wsl$\Ubuntu\home\$env:USERNAME\training"
)

$ErrorActionPreference = "Stop"
$wsl = $WslDir
$src = $PSScriptRoot

# Sync scripts + data into WSL (data/ is gitignored; copy the file the sweep needs).
foreach ($f in "train_qlora.py","eval_compare.py","score_run.py","eval_questions.jsonl") {
  Copy-Item "$src\$f" "$wsl\$f" -Force
}
New-Item -ItemType Directory -Force -Path "$wsl\data" | Out-Null
Copy-Item "$src\$Data" "$wsl\$Data" -Force
Write-Host "Synced scripts + $Data to WSL`n"

$results = @()
foreach ($base in $Bases) {
  $blabel = ($base -split "/")[-1]
  foreach ($ep in $Epochs) {
    $lrTag = ($Lr -replace "0\.0*","").TrimStart("0")
    $out = "sweep_${blabel}-e${ep}".ToLower() -replace "[^a-z0-9._-]","-"
    Write-Host "=== TRAIN $base  epochs=$ep lr=$Lr -> $out ===" -ForegroundColor Cyan
    wsl -d Ubuntu -- bash -lc "cd ~/training && ./.venv/bin/python train_qlora.py --data '$Data' --base '$base' --out '$out' --epochs $ep --lr $Lr 2>&1 | grep -E 'train_loss|Adapter saved|OutOfMemory|CUDA out of memory|Error'"

    Write-Host "--- VOICE eval $out (base-vs-adapter) ---"
    wsl -d Ubuntu -- bash -lc "cd ~/training && ./.venv/bin/python eval_compare.py --adapter '$out' --base '$base' --which adapter --tag voice --max-new $MaxNew > /dev/null 2>&1"
    $score = wsl -d Ubuntu -- bash -lc "cd ~/training && ./.venv/bin/python score_run.py --answers 'answers_${out}.jsonl' --tag voice --json 2>/dev/null"
    try { $rate = ($score | ConvertFrom-Json).pass_rate } catch { $rate = "ERR" }
    Write-Host ("  voice pass_rate = {0}" -f $rate) -ForegroundColor Yellow
    $results += [pscustomobject]@{ adapter = $out; base = $base; epochs = $ep; voice_pass = $rate }
  }
}

Write-Host "`n==== SWEEP LEADERBOARD (voice) ====" -ForegroundColor Green
$results | Sort-Object voice_pass -Descending | Format-Table -AutoSize
Write-Host "Next: council_judge.py over the top answers_*.jsonl, then pick the winner."
