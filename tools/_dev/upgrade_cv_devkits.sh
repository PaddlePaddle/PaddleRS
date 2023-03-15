#!/usr/bin/env bash

DL_DIR="/tmp/_paddlers_cv_repos"
MODELS_DIR="paddlers/models"
IMPORT_PREFIX="paddlers.models."

mkdir -p "${DL_DIR}"
trap "rm -rf ${DL_DIR}" EXIT

function update_apis_from_remote_repo() {
    local src_mod_name="$1"
    local repo_url="$2"
    
    local dst_mod_name="${src_mod_name}"
    local dl_path="${DL_DIR}/${dst_mod_name}"
    local src_mod_path="${dl_path}/${src_mod_name}"
    local dst_mod_path="${MODELS_DIR}/${dst_mod_name}"
    # Clone default branch
    git clone --depth 1 "${repo_url}" "${dl_path}"
    if [ -d "${src_mod_path}" ]; then
        if [ -d "${dst_mod_path}" ]; then
            echo ""
            read -p "${dst_mod_path} already exists. Do you want to remove it?" yes_or_no
            if [ "${yes_or_no}" = 'yes' ]; then
                rm -rf "${dst_mod_path}"
            else
                return 0
            fi
        fi
    else
        return 1
    fi

    cp -r "${src_mod_path}" "${dst_mod_path}"

    # Record commit id
    git -C "${dl_path}" rev-parse HEAD > "${dst_mod_path}/hash.txt"
}

function update_import_statements() {
    local mod_name="$1"
    for f in $(find "${MODELS_DIR}/${mod_name}" -type f -name '*.py'); do
        sed -i -E "s/^([\s]*)import ${mod_name}/\1import ${IMPORT_PREFIX}${mod_name}/g" "$f"
        sed -i -E "s/from ${mod_name}(.*) import/from ${IMPORT_PREFIX}${mod_name}\1 import/g" "$f"
    done
}

update_apis_from_remote_repo 'ppcls' https://github.com/PaddlePaddle/PaddleClas.git
# For paddleclas, we trim the `configs` directory
if [ -d "${MODELS_DIR}/ppcls/configs" ]; then
    rm -rf "${MODELS_DIR}/ppcls/configs"
fi
update_import_statements 'ppcls'

update_apis_from_remote_repo 'ppdet' https://github.com/PaddlePaddle/PaddleDetection.git
update_import_statements 'ppdet'

update_apis_from_remote_repo 'paddleseg' https://github.com/PaddlePaddle/PaddleSeg.git
update_import_statements 'paddleseg'

update_apis_from_remote_repo 'ppgan' https://github.com/PaddlePaddle/PaddleGAN.git
update_import_statements 'ppgan'
