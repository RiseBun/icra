#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/icra2027_4d}"
mkdir -p "$ROOT"/{third_party,data/raw,data/features,models,baselines,configs,scripts,results,logs}

# Existing local repositories are linked read-only by convention. The script does
# not alter their contents and records external repositories at clone time.
ln -sfn "$HOME/VGGT-Omega" "$ROOT/third_party/VGGT-Omega"
ln -sfn "$HOME/VGGT4D" "$ROOT/third_party/VGGT4D"

clone_if_missing() {
  local name="$1" url="$2"
  if [[ ! -e "$ROOT/third_party/$name" ]]; then
    git clone "$url" "$ROOT/third_party/$name"
  fi
  git -C "$ROOT/third_party/$name" rev-parse HEAD > "$ROOT/third_party/${name}.commit"
}

clone_if_missing where2act https://github.com/daerduoCarey/where2act.git
clone_if_missing diffusion_policy https://github.com/real-stanford/diffusion_policy.git
clone_if_missing OpenSceneFlow https://github.com/KTH-RPL/OpenSceneFlow.git

if [[ -d "$HOME/GAF" ]]; then
  ln -sfn "$HOME/GAF" "$ROOT/third_party/GAF"
fi

echo "Prepared $ROOT"
find "$ROOT/third_party" -maxdepth 1 -mindepth 1 -printf '%f\n' | sort
