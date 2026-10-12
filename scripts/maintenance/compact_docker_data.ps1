# One-shot administrator worker. The caller stops/restores the recorded services.
[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunDirectory)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$run = [IO.Path]::GetFullPath($RunDirectory).TrimEnd('\')
if ($workspace -ne 'D:\CoopTrack_Audit' -or
    $run -notmatch '^D:\\CoopTrack_Audit\\reports\\budget_free\\docker_compaction_[a-f0-9]{32}$') {
    throw 'Unexpected workspace/run directory.'
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not ([Security.Principal.WindowsPrincipal]::new($identity)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator rights required.'
}
$disk = 'C:\Users\yujie\AppData\Local\Docker\wsl\disk\docker_data.vhdx'
$record = [ordered]@{ status='WAITING_FOR_OFFLINE'; started_utc=[DateTime]::UtcNow.ToString('o'); disk=$disk; administrator=$true }
function Save-Result {
    [IO.File]::WriteAllText((Join-Path $run 'result.json'),($record | ConvertTo-Json -Depth 8),(New-Object Text.UTF8Encoding($false)))
}
try {
    if (-not [IO.Directory]::Exists($run)) { throw 'Run directory missing.' }
    if (([IO.File]::GetAttributes($run) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Run directory is a reparse point.' }
    Save-Result
    [IO.File]::WriteAllText((Join-Path $run 'ready.txt'),'ADMIN_READY')
    $deadline = [DateTime]::UtcNow.AddMinutes(10)
    while (-not [IO.File]::Exists((Join-Path $run 'offline_ready.txt'))) {
        if ([IO.File]::Exists((Join-Path $run 'abort.txt'))) { throw 'Caller aborted before compaction.' }
        if ([DateTime]::UtcNow -gt $deadline) { throw 'Offline gate timed out; disk untouched.' }
        Start-Sleep -Seconds 2
    }
    if (Get-Process 'Docker Desktop','com.docker.backend' -ErrorAction SilentlyContinue) { throw 'Docker Desktop/backend still running.' }
    $running = (& wsl --list --running --quiet | Out-String).Replace([string][char]0,'')
    if ($LASTEXITCODE -ne 0 -or $running -match '(?m)^\s*docker-desktop\s*$') { throw 'Docker WSL is still running or cannot be checked.' }
    $cursor = $disk
    while ($cursor) {
        if (([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'Disk path contains a reparse point.' }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    $handle = [IO.File]::Open($disk,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None)
    $handle.Dispose()
    $record.before_bytes = ([IO.FileInfo]::new($disk)).Length
    $record.c_free_before = ([IO.DriveInfo]::new('C:\')).AvailableFreeSpace
    $record.status = 'COMPACTING'; Save-Result
    $inputPath = Join-Path $run 'compact.diskpart.txt'
    [IO.File]::WriteAllText($inputPath,"select vdisk file=`"$disk`"`r`ncompact vdisk`r`nexit`r`n",[Text.Encoding]::ASCII)
    & "$env:WINDIR\System32\diskpart.exe" /s $inputPath 2>&1 | Out-File -LiteralPath (Join-Path $run 'diskpart.log') -Encoding utf8
    $record.diskpart_exit_code = $LASTEXITCODE
    $record.after_bytes = ([IO.FileInfo]::new($disk)).Length
    $record.c_free_after = ([IO.DriveInfo]::new('C:\')).AvailableFreeSpace
    $record.vhdx_bytes_reclaimed = $record.before_bytes - $record.after_bytes
    $log = Get-Content -LiteralPath (Join-Path $run 'diskpart.log') -Raw -Encoding UTF8
    if ($record.diskpart_exit_code -ne 0 -or $record.after_bytes -gt $record.before_bytes) { throw 'DiskPart failed or VHDX grew.' }
    if ($record.vhdx_bytes_reclaimed -gt 0) { $record.status='COMPLETE' }
    elseif ($log -match '(?i)successfully compacted|\u6210\u529f\u538b\u7f29') { $record.status='COMPLETE_NO_RECLAIM' }
    else { throw 'No physical reduction or confirmed DiskPart success; inspect diskpart.log.' }
} catch {
    $record.status='FAILED'; $record.error=$_.Exception.Message
} finally {
    $record.finished_utc=[DateTime]::UtcNow.ToString('o'); Save-Result
}
if ($record.status -notin @('COMPLETE','COMPLETE_NO_RECLAIM')) { exit 1 }
