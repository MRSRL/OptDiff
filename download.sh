#!/usr/bin/env bash
# Download the pretrained diffusion models into checkpoints/<dataset>/ and check them against the files behind the
# paper's runs. The dataset configs (configs/mri/fastmri.yaml, configs/natimg/imagenet.yaml, ffhq.yaml) point there.
#
#   bash download.sh                 # all models
#   bash download.sh imagenet ffhq   # some of them: fastmri, imagenet, ffhq
#
# The test data are prepared from the original datasets with scripts/prepare_fastmri.py and scripts/prepare_natimg.py.
set -euo pipefail
cd "$(dirname "$0")"  # repository root

check() {  # check <md5> <file>
    echo "$1  $2" | md5sum -c - || { echo "Checksum mismatch: $2" >&2; exit 1; }
}

have() {  # have <md5> <file>: true if the file was already downloaded
    [ -f "$2" ] && echo "$1  $2" | md5sum -c --status - && echo "$2: already downloaded"
}

gdown_cmd() {  # gdown if installed, otherwise through uvx
    if command -v gdown > /dev/null; then
        gdown "$@"
    elif command -v uvx > /dev/null; then
        uvx gdown "$@"
    else
        echo "Downloading from Google Drive needs gdown (pip install gdown)" >&2
        exit 1
    fi
}

fastmri() {
    # EDM prior trained on fully sampled fastMRI brain data: the supervised (R=1) model of ambient-diffusion-mri
    # (https://github.com/utcsilab/ambient-diffusion-mri), taken from its model archive (1.7 GB)
    local ckpt=checkpoints/fastmri/network-snapshot.pkl md5=21086d72c6ffa6bdedd5c36163b76eb3
    if ! have $md5 $ckpt; then
        mkdir -p checkpoints/fastmri
        wget -O checkpoints/fastmri/ambient_models.zip -L \
            https://utexas.box.com/shared/static/axofnwib9kukdpa92ge4ays87dmuvpf7.zip
        unzip -j -o checkpoints/fastmri/ambient_models.zip 'edm/supervised_R=1/network-snapshot.pkl' -d checkpoints/fastmri
        rm checkpoints/fastmri/ambient_models.zip
        check $md5 $ckpt
    fi
}

imagenet() {
    # Unconditional 256x256 ImageNet model of guided-diffusion (https://github.com/openai/guided-diffusion)
    local ckpt=checkpoints/imagenet/256x256_diffusion_uncond.pt md5=fd9dd2335b8736d521de0aed54bd90ca
    if ! have $md5 $ckpt; then
        mkdir -p checkpoints/imagenet
        wget -O $ckpt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
        check $md5 $ckpt
    fi
}

ffhq() {
    # 256x256 FFHQ model of P2 weighting (https://github.com/jychoi118/P2-weighting), from its Google Drive folder
    local ckpt=checkpoints/ffhq/ffhq_p2.pt md5=c7f1939119cf1a41a098d92917ce8fc6
    if ! have $md5 $ckpt; then
        mkdir -p checkpoints/ffhq
        gdown_cmd 14ACthNkradJWBAL0th6z0UguzR6QjFKH -O $ckpt
        check $md5 $ckpt
    fi
}

targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
    targets=(fastmri imagenet ffhq)
fi
for target in "${targets[@]}"; do
    case $target in
        fastmri|imagenet|ffhq) "$target" ;;
        *) echo "Unknown model: $target (fastmri, imagenet or ffhq)" >&2; exit 1 ;;
    esac
done
