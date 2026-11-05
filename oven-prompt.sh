#!/bin/zsh

set -o pipefail

if [[ -z "$1" ]]; then
  echo "Error: Please provide a prompt in quotes." >&2
  echo "Usage: $0 \"Your prompt here\"" >&2
  exit 1
fi

payload=$(jq -n --arg prompt "$1" '{prompt: $prompt}')

curl -sS http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d "$payload" | jq -r '.response'
