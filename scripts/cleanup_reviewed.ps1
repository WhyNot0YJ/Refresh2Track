# Exact reviewed cleanup only. Default mode does not delete or move anything.
[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$ResultPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot)).TrimEnd('\')
$expectedRoot = 'D:\CoopTrack_Audit'
$evidenceRelative = 'audits/project_structure_cleanup/cleanup_evidence.json'
$inventoryRelative = 'audits/project_structure_cleanup/inventory_after.csv.gz'
$evidenceHash = 'd4bd8a1f1c5b2d563a32d82ad8630b39306615af1562cf02df07e873c286652b'
$inventoryHash = '00d68d4912fb266af358e1399526daa11d8fe97734c1d41c9d87ca840a98ad4e'
$safeResultPath = $null
$result = [ordered]@{
    mode = $(if ($Apply) { 'APPLY' } else { 'DRY_RUN' })
    preflight = 'FAIL'; execution = 'NOT_STARTED'; errors = @(); missing = @()
    planned_files = 116; planned_directories = 232; planned_bytes = 2056555438L
    files_deleted = 0; directories_removed = 0; bytes_reclaimed = 0L
    deleted_paths = @(); removed_directory_paths = @()
    protected_files_checked = 0; checkpoint_hashes = 'NOT_CHECKED'
    protected_hashes = 'NOT_CHECKED'; runtime_imports = 'NOT_CHECKED'
    process_check = $null; final = 'PARTIAL'
}

function Safe-Path([string]$relative) {
    if ([string]::IsNullOrWhiteSpace($relative) -or [IO.Path]::IsPathRooted($relative) -or
        $relative.Contains('\') -or $relative -match '(^|/)\.\.?(/|$)' -or $relative.Contains(':')) {
        throw "Unsafe relative path: $relative"
    }
    $absolute = [IO.Path]::GetFullPath((Join-Path $root $relative))
    if (-not $absolute.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside project: $absolute"
    }
    $cursor = $absolute
    while ($cursor -and $cursor.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Junction/symlink/reparse path is prohibited: $cursor"
            }
        }
        if ($cursor -eq $root) { break }
        $cursor = Split-Path -Parent $cursor
    }
    return $absolute
}

function Hash-File([string]$path) {
    # Read-only sharing permits an editor's write-capable handle; refuse a file that actually changes during hashing.
    $before = Get-Item -LiteralPath $path -Force
    $beforeSize = $before.Length; $beforeTime = $before.LastWriteTimeUtc.Ticks
    $stream = [IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,
        ([IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete))
    $hasher = [Security.Cryptography.SHA256]::Create()
    try { $digest = [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-','').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
    $after = Get-Item -LiteralPath $path -Force
    if ($after.Length -ne $beforeSize -or $after.LastWriteTimeUtc.Ticks -ne $beforeTime) {
        throw "File changed while hashing: $path"
    }
    return $digest
}

function Assert-Hash([string]$relative, [string]$expected) {
    $path = Safe-Path $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Protected file missing: $relative" }
    $actual = Hash-File $path
    if ($actual -ne $expected.ToLowerInvariant()) { throw "SHA256 mismatch: $relative" }
    return $actual
}

function Read-ReviewedInventory($whitelist) {
    $stream = [IO.File]::OpenRead((Safe-Path $inventoryRelative))
    $gzip = New-Object IO.Compression.GZipStream($stream, [IO.Compression.CompressionMode]::Decompress)
    $reader = New-Object IO.StreamReader($gzip, [Text.Encoding]::UTF8)
    $found = @{}
    try {
        $header = $reader.ReadLine().Split(',')
        while (($line = $reader.ReadLine()) -ne $null) {
            $separator = $line.IndexOf(',')
            if ($separator -lt 1) { continue }
            $key = $line.Substring(0, $separator)
            # Reviewed target names have no CSV quoting; other columns are parsed by the CSV parser.
            if ($whitelist.Contains($key)) {
                if ($found.ContainsKey($key)) { throw "Duplicate inventory row: $key" }
                $found[$key] = $line | ConvertFrom-Csv -Header $header
            }
        }
    } finally { $reader.Dispose(); $gzip.Dispose(); $stream.Dispose() }
    if ($found.Count -ne $whitelist.Count) { throw 'Inventory does not cover every reviewed target.' }
    return $found
}

function Check-Processes {
    $commandTargets = @($allTargets | ForEach-Object {
        [pscustomobject]@{ relative = $_.path; absolute = (Safe-Path $_.path); slashAbsolute = ($root.Replace('\','/') + '/' + $_.path) }
    })
    $windows = @(Get-CimInstance Win32_Process)
    foreach ($process in $windows) {
        if ($process.ProcessId -eq $PID) { continue }
        $command = [string]$process.CommandLine
        foreach ($target in $commandTargets) {
            if ($command.IndexOf($target.absolute, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
                $command.IndexOf($target.slashAbsolute, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw "Target referenced by Windows PID $($process.ProcessId): $($target.relative)"
            }
        }
        if ($process.Name -match '^(ninja|nvcc|cmake|make|cl)\.exe$' -or
            ($command -match 'mmdetection3d' -and $command -match 'setup\.py.*(build|develop|install)')) {
            throw "Build process present; postpone cleanup: PID $($process.ProcessId) $($process.Name)"
        }
    }
    # Read-only probe. It never removes files, writes outputs, or starts/stops experiments.
    $probe = @'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
plan = json.loads((root / 'audits/project_structure_cleanup/cleanup_evidence.json').read_text())
targets = [(str(root / e['path']), False) for e in plan['merged_files'] + plan['deleted_files']]
targets += [(str(root / e['path']), True) for e in plan['removed_directories']]
hits, errors, commands = [], [], []
def check(pid, value, kind):
    value = value.removesuffix(' (deleted)') if hasattr(value, 'removesuffix') else value.replace(' (deleted)', '')
    for target, isdir in targets:
        if value == target or (isdir and value.startswith(target + '/')):
            hits.append({'pid': pid, 'kind': kind, 'target': target, 'value': value})
for process in pathlib.Path('/proc').iterdir():
    if not process.name.isdigit() or int(process.name) == os.getpid(): continue
    try:
        args = (process / 'cmdline').read_bytes().split(b'\0')
        command = ' '.join(x.decode(errors='replace') for x in args if x)
        if str(root) in command: commands.append({'pid': process.name, 'command': command})
        for arg in args:
            if arg:
                value = arg.decode(errors='replace')
                check(process.name, value, 'cmdline')
                if value.startswith('--') and '=' in value: check(process.name, value.split('=', 1)[1], 'cmdline-option')
        if 'setup.py' in command and any(x in command for x in ('build_ext', 'develop', 'install')):
            errors.append({'pid': process.name, 'error': 'active native build command'})
        for link in [process / 'cwd'] + list((process / 'fd').iterdir()):
            try: check(process.name, os.readlink(str(link)), 'cwd/fd')
            except FileNotFoundError: pass
        for line in (process / 'maps').read_text().splitlines():
            fields = line.split(None, 5)
            if len(fields) == 6 and fields[5].startswith('/'): check(process.name, fields[5], 'maps')
    except FileNotFoundError: pass  # A process or descriptor may exit while being inspected.
    except PermissionError as error: errors.append({'pid': process.name, 'error': str(error)})
print(json.dumps({'hits': hits, 'errors': errors, 'project_processes': commands}))
sys.exit(1 if hits or errors else 0)
'@
    $containerIds = @(& docker ps -q)
    if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect Docker processes; no deletion is permitted.' }
    $checks = @()
    foreach ($containerId in $containerIds) {
        $inspection = (@(& docker inspect $containerId) -join "`n")
        if ($LASTEXITCODE -ne 0) { throw 'Docker mount inspection failed.' }
        $container = ($inspection | ConvertFrom-Json)[0]
        $mount = @($container.Mounts | Where-Object { $_.Type -eq 'bind' -and $_.Source.TrimEnd('\','/') -eq $root })
        if ($mount.Count -eq 0) {
            if (@($container.Mounts | Where-Object { $_.Source -like "$root*" }).Count -gt 0) {
                throw "Container has partial project mounts without a full inspectable root: $containerId"
            }
            continue
        }
        $output = & docker exec -e PYTHONDONTWRITEBYTECODE=1 $containerId python -B -c $probe $mount[0].Destination
        if ($LASTEXITCODE -ne 0) { throw "Container process probe failed: $containerId $output" }
        $checks += [pscustomobject]@{ container = $container.Name; probe = ($output | ConvertFrom-Json) }
    }
    # Detect Windows open handles/loaded DLLs on each exact file. No write access is requested.
    foreach ($entry in $files) {
        $path = Safe-Path $entry.path
        $handle = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
        $handle.Dispose()
    }
    return [pscustomobject]@{ windows_processes_checked = $windows.Count; containers = $checks; status = 'PASS' }
}

function Check-RuntimeImports {
    $code = "import importlib,sys;sys.path[:0]=['/mnt/d/CoopTrack_Audit/scripts','/mnt/d/CoopTrack_Audit/code/CoopTrack'];names=['torch','numpy','mmcv','mmdet','mmdet3d','mmdet3d.ops','projects.mmdet3d_plugin','main_runtime','main_message','main_scheduler','main_eval','main_train','v2_runtime','v2_message','v2_scheduler','v2_eval','v2_train','v2_run'];[importlib.import_module(n) for n in names];print('RUNTIME_IMPORTS_PASS '+str(len(names)))"
    $output = & docker exec -e PYTHONDONTWRITEBYTECODE=1 cooptrack-main python -B -c $code
    if ($LASTEXITCODE -ne 0) { throw "Runtime import verification failed: $output" }
    return [string]($output -join "`n")
}

try {
    if ($root -ne $expectedRoot) { throw "Unexpected project root: $root" }
    if ($ResultPath -and [IO.Path]::GetFullPath($ResultPath).StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Optional machine result must be outside the project; no additional project reports are created.'
    }
    if ($ResultPath) { $safeResultPath = [IO.Path]::GetFullPath($ResultPath) }
    $null = Assert-Hash $evidenceRelative $evidenceHash
    $null = Assert-Hash $inventoryRelative $inventoryHash
    $plan = Get-Content -LiteralPath (Safe-Path $evidenceRelative) -Encoding UTF8 -Raw | ConvertFrom-Json
    if ($plan.root -ne $root) { throw 'Evidence belongs to a different project.' }
    $files = @($plan.merged_files) + @($plan.deleted_files)
    $directories = @($plan.removed_directories)
    $allTargets = @($files) + @($directories)
    $whitelist = New-Object 'Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in $allTargets) {
        if (-not $whitelist.Add([string]$entry.path)) { throw "Duplicate allowlist entry: $($entry.path)" }
        $null = Safe-Path $entry.path
        if ($entry.path -match '(^|/)\.git(/|$)|^experiments/v2/ablations/|^datasets/|^data_infos/|^query_cache/|^checkpoints/') {
            throw "Protected target prohibited: $($entry.path)"
        }
    }
    $parts = @($files | Where-Object { $_.path -match '^data/wheels/torch-cu111-parts/part-\d+-\d+$' })
    $copies = @($plan.merged_files | Where-Object { $_.path -match '^code/mmdetection3d/build/lib[^/]+/.+\.so$' })
    $intermediates = @($plan.deleted_files | Where-Object { $_.path.StartsWith('code/mmdetection3d/build/') })
    $bytecode = @($files | Where-Object { $_.path -match '^scripts/__pycache__/[^/]+\.pyc$' })
    $probes = @($files | Where-Object { $_.path -eq 'data/wheels/torch_range_probe.part' })
    $totalBytes = [long](($files | Measure-Object -Property size -Sum).Sum)
    if ($files.Count -ne 116 -or $directories.Count -ne 232 -or $parts.Count -ne 61 -or $copies.Count -ne 11 -or
        $intermediates.Count -ne 36 -or $bytecode.Count -ne 7 -or $probes.Count -ne 1 -or $totalBytes -ne 2056555438L) {
        throw 'Reviewed counts, categories, or total bytes changed.'
    }
    $inventory = Read-ReviewedInventory $whitelist
    foreach ($entry in $allTargets) {
        $path = Safe-Path $entry.path
        if (-not (Test-Path -LiteralPath $path)) { $result.missing += $entry.path; Write-Output "MISSING $($entry.path)"; continue }
        $item = Get-Item -LiteralPath $path -Force
        $expectedKind = $(if ($files -contains $entry) { 'file' } else { 'directory' })
        if ($inventory[$entry.path].kind -ne $expectedKind -or $item.PSIsContainer -ne ($expectedKind -eq 'directory')) {
            throw "Target type differs from inventory: $($entry.path)"
        }
        if ($expectedKind -eq 'file' -and ($item.Length -ne [long]$entry.size -or $item.Length -ne [long]$inventory[$entry.path].size)) {
            throw "Target size changed: $($entry.path)"
        }
    }
    if ($result.missing.Count) { throw 'Reviewed items are missing; recorded explicitly. No alternate paths or deletions attempted.' }
    foreach ($entry in $directories) {
        foreach ($child in @(Get-ChildItem -LiteralPath (Safe-Path $entry.path) -Force)) {
            $relative = $child.FullName.Substring($root.Length + 1).Replace('\','/')
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0 -or -not $whitelist.Contains($relative)) {
                throw "Unreviewed child inside proposed directory: $relative"
            }
        }
    }
    foreach ($copy in $copies) {
        if ($whitelist.Contains([string]$copy.target)) { throw 'Retained runtime is on the deletion list.' }
        $null = Assert-Hash $copy.path $copy.sha256
        $null = Assert-Hash $copy.target $copy.sha256
    }
    $proof = $plan.environment_before.wheel_fragments.concat_proof
    if ($whitelist.Contains([string]$proof.complete_wheel)) { throw 'Complete wheel is on the deletion list.' }
    $wheelHash = Assert-Hash $proof.complete_wheel $proof.recorded_sha256
    $hasher = [Security.Cryptography.SHA256]::Create()
    $buffer = New-Object byte[] (8 * 1024 * 1024)
    $offset = 0L
    try {
        foreach ($part in @($parts | Sort-Object { [long]([IO.Path]::GetFileName($_.path).Split('-')[1]) })) {
            $range = [IO.Path]::GetFileName($part.path).Split('-')
            if ([long]$range[1] -ne $offset -or [long]$part.size -ne ([long]$range[2] - [long]$range[1] + 1)) {
                throw "Discontinuous wheel fragment: $($part.path)"
            }
            $stream = [IO.File]::OpenRead((Safe-Path $part.path))
            try { while (($read = $stream.Read($buffer,0,$buffer.Length)) -gt 0) { $null = $hasher.TransformBlock($buffer,0,$read,$buffer,0); $offset += $read } }
            finally { $stream.Dispose() }
        }
        $null = $hasher.TransformFinalBlock((New-Object byte[] 0),0,0)
        $combined = [BitConverter]::ToString($hasher.Hash).Replace('-','').ToLowerInvariant()
        if ($combined -ne $wheelHash -or $offset -ne (Get-Item -LiteralPath (Safe-Path $proof.complete_wheel)).Length) {
            throw 'Fragment aggregate is not identical to retained complete wheel.'
        }
    } finally { $hasher.Dispose() }
    $null = Assert-Hash $proof.probe_duplicate.probe $proof.probe_duplicate.probe_sha256
    # Frozen bindings are required. Other protected content is compared with its current pre-operation bytes.
    foreach ($property in $plan.validation_before.source_hashes.PSObject.Properties) {
        $null = Assert-Hash $property.Name $property.Value.expected_sha256
    }
    foreach ($property in $plan.validation_before.checkpoint_hashes.PSObject.Properties) {
        $null = Assert-Hash $property.Name $property.Value.actual_sha256
    }
    $result.checkpoint_hashes = 'PASS'
    Write-Output 'PREFLIGHT Frozen sources and 10 checkpoints: PASS'
    $protected = @{}
    foreach ($property in $plan.protected_hashes_before.PSObject.Properties) {
        # Existing experiment output is mutable and never on the whitelist. Do not reinterpret old state as immutable.
        if ($property.Name.StartsWith('experiments/v2/ablations/') -or $property.Name -eq 'reports/PROJECT_STRUCTURE_CLEANUP.md') { continue }
        if ($whitelist.Contains($property.Name)) { throw "Protected asset in deletion list: $($property.Name)" }
        $protected[$property.Name] = Hash-File (Safe-Path $property.Name)
    }
    $protected[$proof.complete_wheel] = $wheelHash
    foreach ($property in $plan.validation_before.checkpoint_hashes.PSObject.Properties) { $protected[$property.Name] = $property.Value.actual_sha256 }
    $result.protected_files_checked = $protected.Count
    $result.protected_hashes = 'BASELINE_RECORDED'
    Write-Output "PREFLIGHT Protected baseline: $($protected.Count) files"
    $result.process_check = Check-Processes
    Write-Output 'PREFLIGHT Process usage: PASS'
    $result.runtime_imports = Check-RuntimeImports
    # Capture every exact target hash/mtime immediately before any mutation.
    $targetHashes = @{}; $targetTimes = @{}
    foreach ($entry in $files) {
        $path = Safe-Path $entry.path
        $targetHashes[$entry.path] = Hash-File $path
        $targetTimes[$entry.path] = (Get-Item -LiteralPath $path).LastWriteTimeUtc.Ticks
        Write-Output ("{0}_FILE {1} B {2}" -f $result.mode,$entry.size,$entry.path)
    }
    foreach ($entry in $directories) { Write-Output ("{0}_DIRECTORY {1}" -f $result.mode,$entry.path) }
    $result.preflight = 'PASS'
    if (-not $Apply) {
        $result.execution = 'DRY_RUN_ONLY'; $result.final = 'PARTIAL'
    } else {
        $result.process_check = Check-Processes
        $result.execution = 'RUNNING'
        foreach ($entry in $files) {
            $path = Safe-Path $entry.path
            $item = Get-Item -LiteralPath $path -Force
            if ($item.Length -ne [long]$entry.size -or $item.LastWriteTimeUtc.Ticks -ne $targetTimes[$entry.path] -or
                (Hash-File $path) -ne $targetHashes[$entry.path]) { throw "Target changed after preflight: $($entry.path)" }
            $handle = [IO.File]::Open($path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::None)
            $handle.Dispose()
            Remove-Item -LiteralPath $path -Force -ErrorAction Stop
            if (Test-Path -LiteralPath $path) { throw "File still exists: $($entry.path)" }
            $result.files_deleted++; $result.bytes_reclaimed += [long]$entry.size
            $result.deleted_paths += $entry.path
        }
        foreach ($entry in @($directories | Sort-Object { $_.path.Length } -Descending)) {
            $path = Safe-Path $entry.path
            if (@(Get-ChildItem -LiteralPath $path -Force).Count -ne 0) { throw "Directory is not empty: $($entry.path)" }
            # The nonrecursive API fails if any child appears after the emptiness check.
            [IO.Directory]::Delete($path,$false)
            if (Test-Path -LiteralPath $path) { throw "Directory still exists: $($entry.path)" }
            $result.directories_removed++; $result.removed_directory_paths += $entry.path
        }
        foreach ($path in $protected.Keys) { $null = Assert-Hash $path $protected[$path] }
        $result.protected_hashes = 'PASS'
        foreach ($entry in $allTargets) { if (Test-Path -LiteralPath (Safe-Path $entry.path)) { throw "Reviewed target remains: $($entry.path)" } }
        $result.runtime_imports = Check-RuntimeImports
        $result.execution = 'PASS'; $result.final = 'COMPLETE'
    }
} catch {
    $result.errors += $_.Exception.Message
    $result.execution = 'FAIL'
    Write-Output ('CLEANUP_ERROR ' + $_.Exception.Message)
} finally {
    $result['completed_at'] = [DateTimeOffset]::UtcNow.ToString('o')
    $json = $result | ConvertTo-Json -Depth 12
    if ($safeResultPath) { [IO.File]::WriteAllText($safeResultPath,$json,(New-Object Text.UTF8Encoding($false))) }
    Write-Output ('CLEANUP_RESULT ' + ($result | ConvertTo-Json -Depth 12 -Compress))
}
if ($result.execution -eq 'FAIL') { exit 1 }
