$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath 'D:\CoopTrack_Audit').Path.TrimEnd('\')
$protected = @(
    'experiments/main', 'reports/MAIN_EXPERIMENT_FINAL.md',
    'reports/STAGE1_FINAL.md', 'reports/STAGE2_FINAL.md', 'reports/STAGE3_FINAL.md',
    'reports/R50_PAPER_ALIGNMENT_FINAL.md', 'datasets', 'data', 'data_infos',
    'query_cache/train', 'query_cache/val', 'checkpoints', 'code/CoopTrack/.git',
    'docker', 'patches'
)
foreach ($relative in $protected) {
    if (-not (Test-Path -LiteralPath (Join-Path $workspace $relative))) {
        throw "Protected path missing: $relative"
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $workspace 'configs/main_experiment_v2/dev_split.json'))) {
    throw 'This one-time reset requires the abandoned internal split to exist.'
}
if (Test-Path -LiteralPath (Join-Path $workspace 'experiments/v2/metrics/FINAL_PROTOCOL_MANIFEST.json')) {
    throw 'Formal experiment already frozen; reset refused.'
}
$protectedFiles = @(Get-ChildItem -LiteralPath (Join-Path $workspace 'reports') -File |
    Where-Object { $_.Name -like '*FINAL.md' })
$protectedFiles += @(Get-ChildItem -LiteralPath (Join-Path $workspace 'scripts') -File |
    Where-Object { $_.Name -like 'main_*.py' -or $_.Name -eq 'run_main.py' })
$protectedFiles += Get-Item -LiteralPath (Join-Path $workspace 'experiments/main/metrics/main_results.csv')
$before = @{}
foreach ($file in $protectedFiles) {
    $before[$file.FullName] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256 -ErrorAction Stop).Hash
}
$scopes = @('experiments/v2', 'reports/v2', 'logs/v2', 'configs/main_experiment_v2')
$files = @()
$directories = @()
Write-Output 'DELETE_PLAN'
foreach ($scope in $scopes) {
    $target = (Resolve-Path -LiteralPath (Join-Path $workspace $scope)).Path
    if ($target -eq $workspace -or -not $target.StartsWith($workspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe target: $target"
    }
    $items = @(Get-Item -LiteralPath $target) + @(Get-ChildItem -LiteralPath $target -Recurse -Force)
    foreach ($item in $items) {
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
            ($item.FullName -ne $target -and -not $item.FullName.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase))) {
            throw "Unsafe child: $($item.FullName)"
        }
        if ($item.PSIsContainer) { $directories += $item.FullName } else { $files += $item.FullName }
    }
    Write-Output ($target + ' [verified old V2 files=' + ($items | Where-Object { -not $_.PSIsContainer }).Count + ']')
}
$bytecode = @(Get-ChildItem -LiteralPath (Join-Path $workspace 'scripts/__pycache__') -Filter 'v2_*.pyc' -File)
foreach ($file in $bytecode) {
    if (($file.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        -not $file.FullName.StartsWith($workspace + '\scripts\__pycache__\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Unsafe bytecode target'
    }
    $files += $file.FullName
    Write-Output $file.FullName
}
$containerProcesses = docker exec cooptrack-main ps -eo pid,args
if ($LASTEXITCODE -ne 0 -or $containerProcesses -match 'python .*CoopTrack_Audit/scripts/v2_(run|train|eval)') {
    throw 'Container check failed or old writer active'
}
$hostProcesses = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'scripts[/\\]v2_(run|train|eval)'
}
if ($hostProcesses) { throw 'Old host writer still active' }
# Delete only verified individual files, then empty directories without recursion.
foreach ($absolute in $files) { Remove-Item -LiteralPath $absolute -Force }
foreach ($absolute in ($directories | Sort-Object Length -Descending)) {
    if (@(Get-ChildItem -LiteralPath $absolute -Force).Count -ne 0) { throw "Directory not empty: $absolute" }
    Remove-Item -LiteralPath $absolute -Force
}
foreach ($relative in @('experiments/v2/checkpoints', 'experiments/v2/metrics',
    'experiments/v2/communication', 'experiments/v2/predictions', 'logs/v2',
    'reports/v2', 'configs/main_experiment_v2')) {
    New-Item -ItemType Directory -Path (Join-Path $workspace $relative) -Force | Out-Null
}
foreach ($relative in $protected) {
    if (-not (Test-Path -LiteralPath (Join-Path $workspace $relative))) { throw "Post-cleanup missing: $relative" }
}
foreach ($filePath in $before.Keys) {
    if ((Get-FileHash -LiteralPath $filePath -Algorithm SHA256 -ErrorAction Stop).Hash -ne $before[$filePath]) {
        throw "V1 changed: $filePath"
    }
}
$v2Sources = @(Get-ChildItem -LiteralPath (Join-Path $workspace 'scripts') -Filter 'v2_*.py' -File)
if ($v2Sources.Count -ne 8) { throw 'V2 Python source inventory mismatch' }
$lines = @('# V2 cleanup and formal protocol reset', '',
    'Status: PASS. Old experiment processes stopped; protected V1 files unchanged.', '',
    'Stopped: Windows Python PID 36396 and container Python PID 1300. Exact command identities were verified. Docker service, image and cooptrack-main container were preserved.', '',
    'Deleted old V2 trees (only verified files and then empty directories):')
$lines += $scopes | ForEach-Object { '- ' + $_ }
$lines += @('- scripts/__pycache__/v2_*.pyc', '',
    'The abandoned split, checkpoints, selection state, predictions, communication, logs, reports, source identities and temporary files were removed. No old measurements are carried into this experiment.', '',
    'Protected paths checked before and after cleanup:')
$lines += $protected | ForEach-Object { '- ' + $_ }
$lines += @('', 'V1_PRESERVED=PASS. SHA256 unchanged for V1 formal reports, main_*.py scripts, run_main.py and main_results.csv. All eight V2 Python source files were preserved.', '',
    'Formal protocol: all 46 official train scenes / 7362 samples; no internal validation, selection or early stopping. Fresh seed-0 MLP and Transformer each train exactly 12 epochs with true multi-scene batch 4. Four locked per-scene budgets each occur three times. Official val (21 scenes / 3316 samples) starts only after both epoch-12 final models and the final protocol manifest are frozen. Historical official-val P1/P2/P4 are reused. New P6/P8/P10/P12/P16/P24/P32/INIT_ONLY and learned B1-B4 are pre-registered.', '')
[IO.File]::WriteAllLines((Join-Path $workspace 'reports/v2/CLEANUP_RESET.md'), $lines, [Text.UTF8Encoding]::new($false))
Write-Output 'OLD_PROCESS_STOPPED=PASS'
Write-Output 'OLD_V2_RESET=PASS'
Write-Output 'V1_PRESERVED=PASS'
Write-Output ('DELETED_FILES=' + $files.Count)
Write-Output ('PROTECTED_HASHED_FILES=' + $before.Count)
