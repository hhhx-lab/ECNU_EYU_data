#!/bin/bash

# Shared storage guard for the ECNU Dataset264 preparation and training jobs.

s2_public_available_kib() {
    df -Pk /public | awk 'NR == 2 {print $4}'
}

s2_assert_public_free_space() {
    local available_kib required_kib
    available_kib=$(s2_public_available_kib)
    required_kib=$((S2_PUBLIC_MIN_FREE_GIB * 1024 * 1024))
    if (( available_kib < required_kib )); then
        echo "Emergency /public free-space guard failed: available_kib=${available_kib}, required_kib=${required_kib}" >&2
        return 1
    fi
}

s2_check_active_storage_free_space() {
    [[ "${S2_STORAGE_MODE}" == "public_emergency" ]] || return 0
    s2_assert_public_free_space
}

s2_normalize_absolute_path() {
    local value="${1%/}"
    [[ "${value}" == /* ]] || {
        echo "Storage path must be absolute: ${1}" >&2
        return 1
    }
    case "${value}/" in
        *'//'*|*'/./'*|*'/../'*)
            echo "Storage path contains a non-canonical component: ${1}" >&2
            return 1
            ;;
    esac
    printf '%s\n' "${value}"
}

s2_configure_storage_policy() {
    : "${PROJECT_ROOT:?PROJECT_ROOT is required}"
    : "${S2_HPC_USER_ROOT:?S2_HPC_USER_ROOT is required}"
    : "${S2_STORAGE_ROOT:?S2_STORAGE_ROOT is required}"

    S2_ALLOW_PUBLIC_EMERGENCY="${S2_ALLOW_PUBLIC_EMERGENCY:-0}"
    S2_PUBLIC_MIN_FREE_GIB="${S2_PUBLIC_MIN_FREE_GIB:-1024}"
    S2_PUBLIC_CHECK_INTERVAL_SECONDS="${S2_PUBLIC_CHECK_INTERVAL_SECONDS:-300}"
    S2_CLEAN_PUBLIC_AFTER_SUCCESS="${S2_CLEAN_PUBLIC_AFTER_SUCCESS:-1}"

    [[ "${S2_ALLOW_PUBLIC_EMERGENCY}" =~ ^[01]$ ]] || {
        echo "S2_ALLOW_PUBLIC_EMERGENCY must be 0 or 1" >&2
        return 1
    }
    [[ "${S2_CLEAN_PUBLIC_AFTER_SUCCESS}" =~ ^[01]$ ]] || {
        echo "S2_CLEAN_PUBLIC_AFTER_SUCCESS must be 0 or 1" >&2
        return 1
    }
    [[ "${S2_PUBLIC_MIN_FREE_GIB}" =~ ^[0-9]+$ ]] && (( S2_PUBLIC_MIN_FREE_GIB >= 1024 )) || {
        echo "S2_PUBLIC_MIN_FREE_GIB must be an integer >= 1024" >&2
        return 1
    }
    [[ "${S2_PUBLIC_CHECK_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] && (( S2_PUBLIC_CHECK_INTERVAL_SECONDS >= 60 )) || {
        echo "S2_PUBLIC_CHECK_INTERVAL_SECONDS must be an integer >= 60" >&2
        return 1
    }

    S2_STORAGE_ROOT=$(s2_normalize_absolute_path "${S2_STORAGE_ROOT}") || return 1
    S2_PUBLIC_EMERGENCY_ROOT=$(s2_normalize_absolute_path "${PROJECT_ROOT%/}/work_space/S2/data/ecnu_completion_emergency") || return 1

    case "${S2_STORAGE_ROOT}" in
        /hpc_stor/*)
            if [[ ! -d "${S2_HPC_USER_ROOT}" || ! -w "${S2_HPC_USER_ROOT}" ]]; then
                echo "Missing writable HPC storage root: ${S2_HPC_USER_ROOT}" >&2
                return 1
            fi
            S2_STORAGE_MODE=hpc
            ;;
        /public/*)
            if [[ "${S2_ALLOW_PUBLIC_EMERGENCY}" != "1" ]]; then
                echo "Dataset264 storage under /public requires S2_ALLOW_PUBLIC_EMERGENCY=1" >&2
                return 1
            fi
            if [[ "${S2_STORAGE_ROOT}" != "${S2_PUBLIC_EMERGENCY_ROOT}" ]]; then
                echo "Emergency /public storage must equal ${S2_PUBLIC_EMERGENCY_ROOT}, got: ${S2_STORAGE_ROOT}" >&2
                return 1
            fi
            s2_assert_public_free_space || return 1
            S2_STORAGE_MODE=public_emergency
            ;;
        *)
            echo "Dataset264 storage must be under /hpc_stor or the approved /public emergency root: ${S2_STORAGE_ROOT}" >&2
            return 1
            ;;
    esac

    export S2_STORAGE_ROOT S2_STORAGE_MODE
    echo "S2_STORAGE_POLICY_PASS mode=${S2_STORAGE_MODE} root=${S2_STORAGE_ROOT}"
    if [[ "${S2_STORAGE_MODE}" == "public_emergency" ]]; then
        echo "S2_PUBLIC_EMERGENCY free_kib=$(s2_public_available_kib) minimum_gib=${S2_PUBLIC_MIN_FREE_GIB}"
    fi
}

s2_start_storage_watchdog() {
    S2_STORAGE_WATCHDOG_PID=""
    [[ "${S2_STORAGE_MODE}" == "public_emergency" ]] || return 0

    local owner_pid=$$
    (
        while kill -0 "${owner_pid}" 2>/dev/null; do
            sleep "${S2_PUBLIC_CHECK_INTERVAL_SECONDS}"
            if ! s2_assert_public_free_space; then
                echo "S2_PUBLIC_EMERGENCY_ABORT: /public crossed the protected free-space threshold" >&2
                if [[ -n "${SLURM_JOB_ID:-}" ]]; then
                    scancel "${SLURM_JOB_ID}" || true
                else
                    kill -TERM "${owner_pid}" || true
                fi
                exit 1
            fi
        done
    ) &
    S2_STORAGE_WATCHDOG_PID=$!
    export S2_STORAGE_WATCHDOG_PID
}

s2_stop_storage_watchdog() {
    if [[ -n "${S2_STORAGE_WATCHDOG_PID:-}" ]]; then
        kill "${S2_STORAGE_WATCHDOG_PID}" 2>/dev/null || true
        wait "${S2_STORAGE_WATCHDOG_PID}" 2>/dev/null || true
        S2_STORAGE_WATCHDOG_PID=""
    fi
}

s2_write_public_emergency_marker() {
    [[ "${S2_STORAGE_MODE}" == "public_emergency" ]] || return 0
    cat > "${S2_STORAGE_ROOT}/.s2_public_emergency_dataset264" <<EOF
dataset=Dataset264_BraTS2026_MET_Completion
storage_root=${S2_STORAGE_ROOT}
EOF
}

s2_require_public_emergency_marker() {
    [[ "${S2_STORAGE_MODE}" == "public_emergency" ]] || return 0
    local marker="${S2_STORAGE_ROOT}/.s2_public_emergency_dataset264"
    [[ -f "${marker}" ]] || {
        echo "Missing emergency cleanup marker: ${marker}" >&2
        return 1
    }
    grep -Fxq 'dataset=Dataset264_BraTS2026_MET_Completion' "${marker}" || {
        echo "Invalid emergency cleanup marker: ${marker}" >&2
        return 1
    }
    grep -Fxq "storage_root=${S2_STORAGE_ROOT}" "${marker}" || {
        echo "Emergency cleanup marker has the wrong storage root: ${marker}" >&2
        return 1
    }
}

s2_cleanup_public_emergency_regenerable_data() {
    [[ "${S2_STORAGE_MODE}" == "public_emergency" ]] || return 0
    [[ "${S2_CLEAN_PUBLIC_AFTER_SUCCESS}" == "1" ]] || {
        echo "S2_PUBLIC_EMERGENCY_CLEANUP skipped by S2_CLEAN_PUBLIC_AFTER_SUCCESS=0"
        return 0
    }
    s2_require_public_emergency_marker

    local dataset=Dataset264_BraTS2026_MET_Completion
    local targets=(
        "${S2_STORAGE_ROOT}/nnUNet_raw/${dataset}/imagesTr"
        "${S2_STORAGE_ROOT}/nnUNet_raw/${dataset}/labelsTr"
        "${S2_STORAGE_ROOT}/nnUNet_raw/${dataset}/imagesTs"
        "${S2_STORAGE_ROOT}/nnUNet_raw/${dataset}/labelsTs"
        "${S2_STORAGE_ROOT}/nnUNet_preprocessed/${dataset}/nnUNetPlans_3d_fullres"
        "${S2_STORAGE_ROOT}/completion_case_folders"
    )
    local target
    for target in "${targets[@]}"; do
        [[ "${target}" == "${S2_STORAGE_ROOT}/"* ]] || {
            echo "Refusing unsafe cleanup target: ${target}" >&2
            return 1
        }
        if [[ -e "${target}" || -L "${target}" ]]; then
            find "${target}" -depth -delete
            echo "S2_PUBLIC_EMERGENCY_CLEANED target=${target}"
        fi
    done
}
