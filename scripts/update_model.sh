#!/bin/bash

set -e

# Variables
NEW_MODEL="dataset/new_model_folder/model.pkl"
REPO_MODEL="dataset/git_repo_FND/model.pkl"
APP_MODEL="Fake_News_App/model.pkl"
GITHUB_RUN_NUMBER="${GITHUB_RUN_NUMBER:-local}"
BRANCH_NAME="feature/new_model_${GITHUB_RUN_NUMBER}"

# Ensure GitHub CLI is authenticated
echo "$GITHUB_TOKEN" | gh auth login --with-token

# Remove existing model files
if [ -f "$REPO_MODEL" ]; then
  rm -f "$REPO_MODEL"
  echo "Removed $REPO_MODEL"
fi

if [ -f "$APP_MODEL" ]; then
  rm -f "$APP_MODEL"
  echo "Removed $APP_MODEL"
fi

# Copy the new model to the required locations
cp "$NEW_MODEL" "$REPO_MODEL"
echo "Copied new model to $REPO_MODEL"

cp "$NEW_MODEL" "$APP_MODEL"
echo "Copied new model to $APP_MODEL"

# Git operations
git checkout -b "$BRANCH_NAME"
git add "$REPO_MODEL" "$APP_MODEL"
git commit -m "Update model.pkl in app and repo folders with new model"
git push origin "$BRANCH_NAME"

# Create a pull request
gh pr create --base master --head "$BRANCH_NAME" --title "Update model.pkl with new model" --body "This PR updates the model.pkl file in the app and repo folders with the new trained model."

echo "Script execution completed successfully."
