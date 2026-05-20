#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/create_github_repo_after_zip.sh sali-pycce-prototype public
# Requires GitHub CLI: https://cli.github.com/

REPO_NAME="${1:-sali-pycce-prototype}"
VISIBILITY="${2:-private}" # private or public

git init
git add .
git commit -m "Initial SALI-PyCCE prototype"

gh repo create "$REPO_NAME" --"$VISIBILITY" --source=. --remote=origin --push
