#!/usr/bin/env bash
#
# One-shot Azure Container Apps deployment for guideline-gpt.
#
# Prereqs: az CLI (`az login` first), an active subscription, and either:
#   - OPENAI_API_KEY in the environment (always required), and optionally
#   - ANTHROPIC_API_KEY in the environment (required if LLM_PROVIDER=anthropic).
#
# Overridable variables (export before running):
#   RG, LOC, APP, ACR_NAME, TAG, LLM_PROVIDER, ANTHROPIC_MODEL, OPENAI_MODEL.

set -euo pipefail

RG="${RG:-rg-guideline-gpt}"
LOC="${LOC:-eastus}"
APP="${APP:-guideline-gpt}"
TAG="${TAG:-latest}"
LLM_PROVIDER="${LLM_PROVIDER:-anthropic}"
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-haiku-4-5-20251001}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"

# ACR names are globally unique, alphanumeric, 5-50 chars. Derive a stable
# default from the resource group so re-runs don't create new registries.
DEFAULT_ACR="acr${APP//-/}$(echo -n "$RG" | sha1sum | cut -c1-6)"
ACR_NAME="${ACR_NAME:-$DEFAULT_ACR}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set (it powers embeddings — required regardless of provider)." >&2
  exit 1
fi

if [[ "$LLM_PROVIDER" == "anthropic" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=> Resource group:   $RG ($LOC)"
echo "=> App / image tag:  $APP:$TAG"
echo "=> ACR:              $ACR_NAME"
echo "=> LLM provider:     $LLM_PROVIDER"
echo

echo "=> [1/4] Ensuring resource group exists…"
az group create --name "$RG" --location "$LOC" --output none

echo "=> [2/4] Ensuring Azure Container Registry exists…"
if ! az acr show --name "$ACR_NAME" --resource-group "$RG" --output none 2>/dev/null; then
  az acr create \
    --resource-group "$RG" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled false \
    --output none
fi

echo "=> [3/4] Building and pushing image with ACR cloud build (~5 min on first run)…"
az acr build \
  --registry "$ACR_NAME" \
  --image "$APP:$TAG" \
  --file "$REPO_ROOT/Dockerfile" \
  "$REPO_ROOT"

echo "=> [4/4] Deploying Bicep template…"
az deployment group create \
  --resource-group "$RG" \
  --template-file "$REPO_ROOT/infra/main.bicep" \
  --parameters \
      appName="$APP" \
      acrName="$ACR_NAME" \
      imageTag="$TAG" \
      llmProvider="$LLM_PROVIDER" \
      anthropicModel="$ANTHROPIC_MODEL" \
      openaiModel="$OPENAI_MODEL" \
      openaiApiKey="$OPENAI_API_KEY" \
      anthropicApiKey="${ANTHROPIC_API_KEY:-}" \
  --output none

URL="$(az containerapp show \
        --name "$APP" \
        --resource-group "$RG" \
        --query properties.configuration.ingress.fqdn \
        --output tsv)"
echo
echo "Deployed: https://$URL"
echo
echo "Notes:"
echo "  - The app starts with an empty corpus. Upload PDFs via the UI and click 'Re-ingest',"
echo "    or 'az containerapp exec' in and run 'guideline-gpt ingest' against a mounted volume."
echo "  - minReplicas=0: the first request after idle has a cold-start delay."
