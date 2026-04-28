#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

directories=(
	"data"
	"data/raw"
	"data/interim"
	"data/processed"
	"data/external"
	"notebooks"
	"src"
	"src/data"
	"src/models"
	"src/training"
	"src/evaluation"
	"scripts"
	"configs"
	"models"
	"models/checkpoints"
	"models/exported"
	"reports"
	"reports/figures"
	"logs"
	"experiments"
)

for directory in "${directories[@]}"; do
	mkdir -p "${project_root}/${directory}"
done

echo "Created project directories under ${project_root}"
