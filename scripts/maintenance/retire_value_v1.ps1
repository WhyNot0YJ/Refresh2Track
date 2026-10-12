# Explicitly authorized H retirement. No recursive deletion or service restart.
[CmdletBinding()]
param([switch]$Apply)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..')).TrimEnd('\')
$whitelistRelative = 'reports/budget_free/H_RETIREMENT_WHITELIST.csv.gz'
$approvedWhitelistSha = '1b87467de1fcc558995ebfa0b7abcdcd7c9618f64095fdd2624e866eaf572fbe'
$manifestPath = Join-Path $root 'reports/budget_free/H_RETIREMENT_MANIFEST.json'
$scopes = @('scripts/budget_free', 'configs/budget_free',
    'experiments/budget_free/value_v1', 'experiments/budget_free/value_diagnostics_v1', 'logs/budget_free')
$result = [ordered]@{
    mode = $(if ($Apply) { 'APPLY' } else { 'DRY_RUN' }); started_utc = [DateTime]::UtcNow.ToString('o')
    preflight = 'NOT_RUN'; execution = 'NOT_STARTED'; status = 'BLOCKED'
    whitelist_sha256 = $approvedWhitelistSha
    files_removed = 0; directories_removed = 0; logical_bytes_reclaimed = 0L
    already_absent = 0; failures = @(); files = @(); directories = @()
    archive_verification = 'NOT_RUN'; protected_verification = 'NOT_RUN'; runtime_imports = $null; process_check = $null
    volume_free_before = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($root))).AvailableFreeSpace
    volume_free_after = $null; remaining = @(); policy_blocked_items = @()
}
$fileResults = New-Object 'Collections.Generic.List[object]'
$directoryResults = New-Object 'Collections.Generic.List[object]'

function Safe-Path([string]$relative, [bool]$deletion = $false) {
    if ([string]::IsNullOrWhiteSpace($relative) -or [IO.Path]::IsPathRooted($relative) -or
        $relative.Contains('\') -or $relative.Contains(':') -or $relative -match '(^|/)\.\.?(/|$)') {
        throw "Unsafe relative path: $relative"
    }
    $absolute = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $absolute.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside project/root target: $absolute"
    }
    if ($deletion -and $relative -notmatch '^(scripts/budget_free|configs/budget_free|logs/budget_free|experiments/budget_free/(value_v1|value_diagnostics_v1))(/|$)|^experiments/budget_free$') {
        throw "Protected or unapproved deletion target: $relative"
    }
    $cursor = $absolute
    while ($cursor.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        if ([IO.File]::Exists($cursor) -or [IO.Directory]::Exists($cursor)) {
            if (([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse/junction/link prohibited: $cursor"
            }
        }
        if ($cursor -eq $root) { break }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}

function Assert-Hash([string]$relative, [string]$expected) {
    $path = Safe-Path $relative
    if (-not [IO.File]::Exists($path)) { throw "Required file missing: $relative" }
    $before = [IO.FileInfo]::new($path)
    $size = $before.Length; $ticks = $before.LastWriteTimeUtc.Ticks
    $stream = [IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,
        ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $actual = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
    $after = [IO.FileInfo]::new($path)
    if ($after.Length -ne $size -or $after.LastWriteTimeUtc.Ticks -ne $ticks -or $actual -ne $expected) {
        throw "Hash/identity changed: $relative"
    }
}

function Run-ReadOnly([string]$program, [string]$arguments, [string]$stdin = '') {
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $program; $info.Arguments = $arguments
    $info.UseShellExecute = $false; $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true; $info.RedirectStandardError = $true
    $info.RedirectStandardInput = $true
    $process = [Diagnostics.Process]::Start($info)
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    if ($stdin) {
        $writeTask = $process.StandardInput.WriteAsync($stdin)
        if (-not $writeTask.Wait(5000)) {
            $clientId = $process.Id; $process.Kill(); $process.WaitForExit(); $process.Dispose()
            return [pscustomobject]@{ status = 'TIMEOUT'; exit_code = $null; stdout = ''; stderr = "$program read-only client $clientId did not consume probe stdin" }
        }
    }
    $process.StandardInput.Close()
    if (-not $process.WaitForExit(15000)) {
        # Only this newly launched read-only client is stopped; never a container/service.
        $clientId = $process.Id; $process.Kill(); $process.WaitForExit(); $process.Dispose()
        return [pscustomobject]@{ status = 'TIMEOUT'; exit_code = $null; stdout = ''; stderr = "$program read-only client $clientId timed out after 15 seconds" }
    }
    $value = [pscustomobject]@{ status = 'RETURNED'; exit_code = $process.ExitCode; stdout = $stdout.Result; stderr = $stderr.Result }
    $process.Dispose(); return $value
}

function Check-Processes {
    $windows = @(Get-CimInstance Win32_Process)
    $hits = @()
    foreach ($process in $windows) {
        if ($process.ProcessId -eq $PID -or $process.Name -notmatch '^(python.*|wsl|bash|sh|cmd|powershell|pwsh)\.exe$') { continue }
        $command = ([string]$process.CommandLine).Replace('\','/')
        foreach ($scope in $scopes) {
            if ($command.IndexOf($scope,[StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $hits += [pscustomobject]@{ pid = $process.ProcessId; scope = $scope; command = $command }
            }
        }
    }
    $snapshot = [ordered]@{ windows_processes_checked = $windows.Count; windows_hits = $hits; docker = $null; containers = @(); wsl = $null; status = 'UNCONFIRMED' }
    $result.process_check = $snapshot
    if ($hits.Count) { throw 'BLOCKED_ACTIVE: a Windows process references an H scope; no process is killed.' }
    $distros = Run-ReadOnly 'wsl' '--list --running --quiet'
    $snapshot.wsl = [ordered]@{ list = $distros; probes = @() }
    $engineProcesses = @($windows | Where-Object { $_.Name -match '^(Docker Desktop|com\.docker\..*|dockerd|wsl|wslhost|wslrelay)\.exe$|^vmmem' })
    if ($distros.status -eq 'RETURNED' -and $distros.exit_code -eq 0 -and
        -not $distros.stdout.Replace([string][char]0,'').Trim() -and -not $engineProcesses.Count) {
        # A stopped local engine and zero running WSL instances establish absence of VM writers.
        $context = Run-ReadOnly 'docker' 'context inspect --format "{{json .Endpoints.docker.Host}}"'
        if ($context.status -eq 'RETURNED' -and $context.exit_code -eq 0 -and $context.stdout -match 'npipe:') {
            $snapshot.docker = [pscustomobject]@{ status = 'OFFLINE'; context = $context; engine_processes = 0 }
            $snapshot.status = 'PASS_LOCAL_ENGINE_OFFLINE_WSL_STOPPED'
            return
        }
    }
    $docker = Run-ReadOnly 'docker' 'ps -q'
    $snapshot.docker = $docker
    if ($docker.status -ne 'RETURNED' -or $docker.exit_code -ne 0) {
        throw 'BLOCKED_ACTIVE: Docker writer visibility unavailable; empty timeout output does not mean no containers.'
    }
    $probe = [IO.File]::ReadAllText((Safe-Path 'scripts/maintenance/check_h_writers.py'))
    foreach ($containerId in @($docker.stdout -split '\s+' | Where-Object { $_ })) {
        if ($containerId -notmatch '^[a-f0-9]{12,64}$') { throw 'Invalid Docker container identity.' }
        $inspection = Run-ReadOnly 'docker' ("inspect $containerId")
        if ($inspection.exit_code -ne 0 -or $inspection.status -ne 'RETURNED') { throw 'Docker mount inspection unavailable.' }
        $inspectedContainers = ConvertFrom-Json -InputObject $inspection.stdout
        if ($inspectedContainers.Count -ne 1) { throw 'Unexpected Docker inspection count.' }
        $container = $inspectedContainers[0]
        $mounts = @($container.Mounts | Where-Object { $_.Type -eq 'bind' -and $_.Source.Replace('\','/') -match '(?i)(^|/)CoopTrack_Audit(/|$)' })
        if (-not $mounts.Count) { continue }
        $mount = @($mounts | Where-Object { $_.Source.Replace('\','/').TrimEnd('/') -match '(?i)/CoopTrack_Audit$' })
        if ($mount.Count -ne 1 -or $mount[0].Destination -notmatch '^/[A-Za-z0-9_./-]+$') { throw 'Project mount cannot be safely inspected.' }
        $response = Run-ReadOnly 'docker' ("exec -i -e PYTHONDONTWRITEBYTECODE=1 $containerId python -B - " + $mount[0].Destination) $probe
        $snapshot.containers += [pscustomobject]@{ id = $containerId; response = $response }
        if ($response.status -ne 'RETURNED' -or $response.exit_code -ne 0) { throw 'BLOCKED_ACTIVE: container H references or process inspection errors.' }
        $probeResult = ConvertFrom-Json -InputObject $response.stdout
        if ($probeResult.status -ne 'PASS') { throw 'Container process probe did not confirm PASS.' }
    }
    # Check every running WSL distro; docker-desktop /proc includes VM container processes.
    if ($distros.status -ne 'RETURNED' -or $distros.exit_code -ne 0) { throw 'BLOCKED_ACTIVE: WSL writer visibility unavailable.' }
    $wslProbe = [IO.File]::ReadAllText((Safe-Path 'scripts/maintenance/check_h_writers.sh')).Replace("`r`n", "`n")
    foreach ($distro in @(($distros.stdout.Replace([string][char]0,'') -split '[\r\n]+') | Where-Object { $_.Trim() })) {
        $distro = $distro.Trim()
        if ($distro -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Unexpected WSL distro name.' }
        $linuxRoot = '/mnt/d/CoopTrack_Audit'
        $arguments = "-d $distro -u root --cd / --exec /bin/sh -s -- $linuxRoot"
        $response = Run-ReadOnly 'wsl' $arguments $wslProbe
        $snapshot.wsl.probes += [pscustomobject]@{ distro = $distro; response = $response }
        if ($response.status -ne 'RETURNED' -or $response.exit_code -ne 0 -or $response.stdout -notmatch '(?m)^H_WRITER_PROC_STATUS=PASS\s*$') { throw 'BLOCKED_ACTIVE: WSL /proc could not establish H writer absence.' }
    }
    $snapshot.status = 'PASS'
}

function Check-ArchiveAndProtected {
    $archive = Get-Content -LiteralPath (Safe-Path 'reports/budget_free/H_ARCHIVE_VERIFICATION.json') -Encoding UTF8 -Raw | ConvertFrom-Json
    if ($archive.status -ne 'PASS' -or -not $archive.archive_ready_for_H_source_deletion) { throw 'Archive not verified.' }
    foreach ($entry in @($archive.archive_outputs) + @($archive.original_reports_unchanged)) { Assert-Hash $entry.source $entry.sha256 }
    $result.archive_verification = 'PASS'
    $protected = Get-Content -LiteralPath (Safe-Path 'reports/budget_free/H_PROTECTED_BASELINE.json') -Encoding UTF8 -Raw | ConvertFrom-Json
    foreach ($entry in $protected.files) { Assert-Hash $entry.path $entry.sha256 }
    $result.protected_verification = 'PASS'
    $runtimePath = Safe-Path 'scripts/maintenance/check_retired_v2_runtime.py'
    $runtime = Run-ReadOnly 'python' ('-B "' + $runtimePath + '"')
    $result.runtime_imports = $runtime
    if ($runtime.status -ne 'RETURNED' -or $runtime.exit_code -ne 0) { throw 'Protected runtime check failed.' }
}

try {
    if ($root -ne 'D:\CoopTrack_Audit') { throw "Unexpected project: $root" }
    Assert-Hash $whitelistRelative $approvedWhitelistSha
    $stream = [IO.File]::OpenRead((Safe-Path $whitelistRelative))
    $gzip = New-Object IO.Compression.GZipStream($stream, [IO.Compression.CompressionMode]::Decompress)
    $reader = New-Object IO.StreamReader($gzip, [Text.Encoding]::UTF8)
    try { $entries = @($reader.ReadToEnd() | ConvertFrom-Csv) }
    finally { $reader.Dispose(); $gzip.Dispose(); $stream.Dispose() }
    $known = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $entries) {
        if (-not $known.Add($entry.path) -or $entry.classification -ne 'DELETE_H_ONLY') { throw 'Invalid/duplicate whitelist entry.' }
        $path = Safe-Path $entry.path $true
        if ($path -ne $entry.absolute_path) { throw 'Resolved path differs from approved path.' }
        if (-not (Test-Path -LiteralPath $path)) { continue }
        $item = Get-Item -LiteralPath $path -Force
        if ($item.PSIsContainer -ne ($entry.kind -eq 'directory')) { throw "Target type changed: $path" }
        if ($entry.kind -eq 'file' -and ($item.Length -ne [long]$entry.bytes -or $item.LastWriteTimeUtc.Ticks -ne [long]$entry.mtime_ticks)) { throw "Target changed: $path" }
    }
    # Only direct children are enumerated. New files/dirs must receive a new reviewed whitelist.
    foreach ($entry in @($entries | Where-Object kind -eq 'directory')) {
        $path = Safe-Path $entry.path $true
        if (-not [IO.Directory]::Exists($path)) { continue }
        foreach ($child in [IO.Directory]::EnumerateFileSystemEntries($path)) {
            $relative = $child.Substring($root.Length + 1).Replace('\','/')
            if (-not $known.Contains($relative)) { throw "Unreviewed new child: $relative" }
        }
    }
    Check-ArchiveAndProtected
    Check-Processes
    foreach ($entry in @($entries | Where-Object kind -eq 'file')) {
        $path = Safe-Path $entry.path $true
        if (-not [IO.File]::Exists($path)) { continue }
        $handle = [IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None)
        $handle.Dispose()
    }
    $result.preflight = 'PASS'
    if (-not $Apply) { $result.status = 'DRY_RUN_PASS' }
    else {
        Check-Processes
        $result.execution = 'RUNNING'
        foreach ($entry in @($entries | Where-Object kind -eq 'file')) {
            $record = [ordered]@{ path = $entry.path; bytes = [long]$entry.bytes; category = $entry.category; status = 'FAILED' }
            try {
                $path = Safe-Path $entry.path $true
                if (-not [IO.File]::Exists($path)) { $record.status = 'ALREADY_ABSENT'; $result.already_absent++ }
                else {
                    $item = [IO.FileInfo]::new($path)
                    if ($item.Length -ne [long]$entry.bytes -or $item.LastWriteTimeUtc.Ticks -ne [long]$entry.mtime_ticks) { throw 'Target changed after preflight.' }
                    # Deny readers/writers while allowing this exact deletion request.
                    $handle = [IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Delete)
                    try { Remove-Item -LiteralPath $path -ErrorAction Stop } finally { $handle.Dispose() }
                    if ([IO.File]::Exists($path)) { throw 'File still exists after deletion.' }
                    $record.status = 'REMOVED'; $result.files_removed++; $result.logical_bytes_reclaimed += [long]$entry.bytes
                }
            } catch { $record.error = $_.Exception.Message; $result.failures += $record }
            $fileResults.Add([pscustomobject]$record)
        }
        foreach ($entry in @($entries | Where-Object kind -eq 'directory' | Sort-Object { $_.path.Length } -Descending)) {
            $record = [ordered]@{ path = $entry.path; status = 'FAILED' }
            try {
                $path = Safe-Path $entry.path $true
                if (-not [IO.Directory]::Exists($path)) { $record.status = 'ALREADY_ABSENT' }
                else { [IO.Directory]::Delete($path,$false); $record.status = 'REMOVED'; $result.directories_removed++ }
            } catch { $record.error = $_.Exception.Message; $result.failures += $record }
            $directoryResults.Add([pscustomobject]$record)
        }
        $result.execution = 'FINISHED'
        Check-ArchiveAndProtected
        $result.remaining = @($entries | Where-Object { Test-Path -LiteralPath (Join-Path $root $_.path) } | ForEach-Object path)
        $result.status = $(if ($result.failures.Count -or $result.remaining.Count) { 'PARTIAL' } else { 'COMPLETE' })
    }
} catch {
    $result.failures += [pscustomobject]@{ stage = 'PREFLIGHT_OR_VERIFICATION'; error = $_.Exception.Message }
    if ($result.files_removed) { $result.status = 'PARTIAL' } else { $result.status = 'BLOCKED' }
    $result.remaining = @([pscustomobject]@{ classification = 'BLOCKED_ACTIVE'; whitelist = $whitelistRelative; reason = $_.Exception.Message })
} finally {
    $result.files = @($fileResults.ToArray()); $result.directories = @($directoryResults.ToArray())
    $result.finished_utc = [DateTime]::UtcNow.ToString('o')
    $result.volume_free_after = ([IO.DriveInfo]::new([IO.Path]::GetPathRoot($root))).AvailableFreeSpace
    $manifest = Get-Content -LiteralPath $manifestPath -Encoding UTF8 -Raw | ConvertFrom-Json
    $history = @()
    if ($manifest.PSObject.Properties.Name -contains 'attempt_history') { $history = @($manifest.attempt_history) }
    $history += [pscustomobject]$result
    $manifest | Add-Member -NotePropertyName attempt_history -NotePropertyValue $history -Force
    if ($Apply) { $manifest | Add-Member -NotePropertyName apply -NotePropertyValue $result -Force }
    else { $manifest | Add-Member -NotePropertyName dry_run -NotePropertyValue $result -Force }
    $manifest.status = $result.status
    [IO.File]::WriteAllText($manifestPath,($manifest | ConvertTo-Json -Depth 35),(New-Object Text.UTF8Encoding($false)))
    Write-Output ("H_METHOD_RETIREMENT mode={0} status={1} files_removed={2} bytes_reclaimed={3}" -f $result.mode,$result.status,$result.files_removed,$result.logical_bytes_reclaimed)
    foreach ($failure in $result.failures) { Write-Output ($failure | ConvertTo-Json -Compress -Depth 8) }
}
if ($result.status -notin @('COMPLETE','DRY_RUN_PASS')) { exit 1 }
