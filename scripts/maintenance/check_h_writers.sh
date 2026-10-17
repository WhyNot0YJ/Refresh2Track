#!/bin/sh
# Read-only /proc probe for minimal WSL distros; stdin: sh -s -- <Linux project root>.
LC_ALL=C
export LC_ALL
root=${1-}
case "$root" in /*) ;; *) printf 'ERROR\tExpected an absolute project root\n'; exit 1 ;; esac
case "$root" in /|*[!A-Za-z0-9_./-]*) printf 'ERROR\tUnsafe project-root spelling\n'; exit 1 ;; esac
[ -d /proc ] && [ -r /proc ] && [ -x /proc ] || { printf 'ERROR\t/proc unavailable\n'; exit 1; }
for tool in tr readlink cat; do
    command -v "$tool" >/dev/null 2>&1 || { printf 'ERROR\tMissing tool %s\n' "$tool"; exit 1; }
done
root=${root%/}
roots=$root
case "$root" in
    /mnt/d/CoopTrack_Audit)
        roots="$roots /mnt/host/d/CoopTrack_Audit /run/desktop/mnt/host/d/CoopTrack_Audit /tmp/docker-desktop-root/mnt/host/d/CoopTrack_Audit"
        ;;
esac
canonical=$(readlink -f "$root" 2>/dev/null)
if [ -n "$canonical" ] && [ "$canonical" != "$root" ]; then roots="$roots $canonical"; fi
scopes='scripts/budget_free configs/budget_free experiments/budget_free/value_v1 experiments/budget_free/value_diagnostics_v1 logs/budget_free'
hits=0
errors=0
races=0
scanned=0

problem() {
    errors=$((errors + 1))
    printf 'ERROR\tpid=%s\tsource=%s\t%s\n' "$pid" "$1" "$2"
}

read_failure() {
    if [ ! -d "$proc" ]; then races=$((races + 1)); return; fi
    case "$2" in
        *'No such file or directory'*|*'No such process'*) races=$((races + 1)) ;;
        *) problem "$1" "$2" ;;
    esac
}

check_path() {
    path=${2% (deleted)}
    case "$path" in /*) ;; *) return ;; esac
    normalized=''
    saved_ifs=$IFS
    IFS=/
    set -f
    for component in $path; do
        case "$component" in
            ''|.) ;;
            ..) normalized=${normalized%/*} ;;
            *) normalized=$normalized/$component ;;
        esac
    done
    set +f
    IFS=$saved_ifs
    path=${normalized:-/}
    for alias in $roots; do
        for scope in $scopes; do
            prefix=$alias/$scope
            case "$path" in
                "$prefix"|"$prefix"/*)
                    hits=$((hits + 1))
                    printf 'HIT\tpid=%s\tscope=%s\tsource=%s\t%s\n' "$pid" "$scope" "$1" "$2"
                    ;;
            esac
        done
    done
}

for proc in /proc/[0-9]*; do
    pid=${proc##*/}
    [ "$pid" = "$$" ] && continue
    [ -d "$proc" ] || { races=$((races + 1)); continue; }
    scanned=$((scanned + 1))
    arguments=$(tr '\000' '\n' < "$proc/cmdline" 2>&1)
    if [ "$?" -ne 0 ]; then read_failure cmdline "$arguments"; arguments=''; fi
    cwd=$(readlink "$proc/cwd" 2>&1)
    if [ "$?" -eq 0 ]; then check_path cwd "$cwd"; else read_failure cwd "$cwd"; cwd=''; fi
    while IFS= read -r argument; do
        candidate=${argument#*=}
        case "$candidate" in
            /*) check_path cmdline "$candidate" ;;
            */*) [ -n "$cwd" ] && check_path cmdline "$cwd/$candidate" ;;
        esac
        for alias in $roots; do
            for scope in $scopes; do
                prefix=$alias/$scope
                case "$argument" in
                    *" $prefix/"*|*"=$prefix/"*|*"'$prefix/"*|*"\"$prefix/"*)
                        hits=$((hits + 1))
                        printf 'HIT\tpid=%s\tscope=%s\tsource=cmdline\t%s\n' "$pid" "$scope" "$argument"
                        ;;
                esac
                if [ "$cwd" = "$alias" ]; then
                    case "$argument" in
                        *" $scope/"*|*"=$scope/"*|*"'$scope/"*|*"\"$scope/"*)
                            hits=$((hits + 1))
                            printf 'HIT\tpid=%s\tscope=%s\tsource=cmdline_relative\t%s\n' "$pid" "$scope" "$argument"
                            ;;
                    esac
                fi
            done
        done
    done <<EOF
$arguments
EOF
    if [ ! -r "$proc/fd" ] || [ ! -x "$proc/fd" ]; then
        if [ -d "$proc" ]; then problem fd 'Cannot enumerate fd directory'; else races=$((races + 1)); fi
    else
        for descriptor in "$proc"/fd/*; do
            [ -L "$descriptor" ] || continue
            value=$(readlink "$descriptor" 2>&1)
            if [ "$?" -eq 0 ]; then
                check_path "fd/${descriptor##*/}" "$value"
            elif [ ! -L "$descriptor" ]; then
                races=$((races + 1))
            else
                read_failure "fd/${descriptor##*/}" "$value"
            fi
        done
    fi
    maps=$(cat "$proc/maps" 2>&1)
    if [ "$?" -ne 0 ]; then read_failure maps "$maps"; maps=''; fi
    while read -r address permissions offset device inode mapped_path; do
        case "$mapped_path" in
            /*) value=$(printf '%b' "$mapped_path"); check_path maps "$value" ;;
        esac
    done <<EOF
$maps
EOF
done
printf 'H_WRITER_PROC_SUMMARY\tprocesses=%s\thits=%s\terrors=%s\tvanished=%s\tignored_pid=%s\n' "$scanned" "$hits" "$errors" "$races" "$$"
if [ "$hits" -gt 0 ] || [ "$errors" -gt 0 ]; then
    printf 'H_WRITER_PROC_STATUS=BLOCKED\n'
    exit 1
fi
printf 'H_WRITER_PROC_STATUS=PASS\n'
