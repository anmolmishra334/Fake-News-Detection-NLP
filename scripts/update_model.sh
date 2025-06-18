#!/bin/bash

# Ensure the script fails on error
set -e

# Define variables
GIT_USERNAME="anmolmishra334"
GIT_PASSWORD="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" 
REPO_NAME="anmolmishra334/Fake-News-Detection-NLP"
NEW_BRANCH_NAME="feature/new_model_$GITHUB_RUN_NUMBER"

# Replace old model files with the new one
rm -f dataset/git_repo_FND/model.pkl
rm -f Fake_News_App/model.pkl
cp dataset/new_model_folder/model.pkl dataset/git_repo_FND/
cp dataset/new_model_folder/model.pkl Fake_News_App/

# Configure git
git config --global user.name "$GIT_USERNAME"
git config --global user.email "$GIT_USERNAME@users.noreply.github.com"

# Add the GitHub credentials to the remote URL
git remote set-url origin https://$GIT_USERNAME:$GIT_PASSWORD@github.com/$REPO_NAME.git

# Create a new branch
git checkout -b $NEW_BRANCH_NAME

# Stage the changes
git add dataset/git_repo_FND/model.pkl Fake_News_App/model.pkl

# Commit the changes
git commit -m "Update model.pkl files for run #$RANDOM"

# Push the changes to the new branch
git push origin $NEW_BRANCH_NAME

# Create a pull request
gh pr create --base master --head $NEW_BRANCH_NAME --title "Update model.pkl files" --body "Automated update for model.pkl files."
