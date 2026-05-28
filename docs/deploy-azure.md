# Deploying to Azure Container Apps

This guide takes you from zero to a public URL serving the inspector UI on
Azure Container Apps (ACA), using a single Bicep template plus a wrapper
script.

> **What you'll provision:** one Resource Group containing an Azure Container
> Registry, Log Analytics workspace, Container Apps environment, Key Vault,
> user-assigned managed identity, and the Container App itself. Estimated
> idle cost: **~$5–$10/month** (see [Cost](#cost) below).

---

## 1. Prerequisites

1. **An Azure subscription.** A [free Azure account](https://azure.microsoft.com/free/)
   gives you $200 of credit for 30 days and 12 months of always-free services
   that more than cover this deployment.
2. **The Azure CLI.** Install: <https://learn.microsoft.com/cli/azure/install-azure-cli>.
3. **bash** (Git Bash, WSL, or any Linux/macOS shell).
4. **An OpenAI API key** in `$OPENAI_API_KEY` (always required — it powers the
   embedding model). Optionally an **Anthropic API key** in
   `$ANTHROPIC_API_KEY` if you'll set `LLM_PROVIDER=anthropic`.

Log in once and pick the subscription:

```bash
az login
az account set --subscription "<your-subscription-id-or-name>"
```

---

## 2. Deploy

From the repository root:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...        # optional, only for provider=anthropic
export LLM_PROVIDER=openai                  # or "anthropic"
bash infra/deploy.sh
```

That's it. The script:

1. Creates the resource group (`rg-guideline-gpt` by default).
2. Creates an Azure Container Registry (Basic SKU) if one doesn't already exist.
3. Builds the Docker image **in the cloud** with `az acr build` — no local
   Docker required — and pushes it to the registry.
4. Deploys `infra/main.bicep`, which provisions Log Analytics, the Container
   Apps environment, Key Vault (with your API keys written as secrets), the
   managed identity (granted `AcrPull` + `Key Vault Secrets User`), and the
   Container App with Key Vault secret references wired into env vars.

When it finishes, you'll see:

```
Deployed: https://<app>.<region>.azurecontainerapps.io
```

Open that URL in a browser and the inspector UI loads.

### Customising

Override defaults via environment variables before running:

| Variable | Default | Effect |
|---|---|---|
| `RG` | `rg-guideline-gpt` | Resource group name |
| `LOC` | `eastus` | Region |
| `APP` | `guideline-gpt` | Container App name and image name |
| `ACR_NAME` | derived | ACR name (must be globally unique) |
| `TAG` | `latest` | Image tag |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model name |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |

---

## 3. After deploy

**Loading documents.** The container starts with no corpus. Two options:

- **Via the UI** — open the deployed URL, expand the *Corpus* bar at the top,
  upload PDFs, and click *Re-ingest*. This stores chunks in the container's
  ephemeral filesystem; they're lost on restart unless you add persistent
  storage (see *Persistence* below).
- **Bake the corpus into the image** — drop PDFs into `documents/` in the
  repo before running `deploy.sh`. The Dockerfile copies them in, and a
  one-time `guideline-gpt ingest` baked into the container as a startup step
  would make the corpus survive restarts. (Not enabled by default; add a
  `RUN guideline-gpt ingest` line to the runtime stage if desired.)

**Persistence.** For production-style persistence, mount an Azure Files share
to `/app/chroma_db` via the Container Apps environment's storage block. The
Bicep here doesn't configure storage by default to keep the demo's cost at
zero when idle.

---

## 4. Cost

The default deployment uses these SKUs:

| Resource | SKU | Notes | Approx monthly cost (idle) |
|---|---|---|---|
| Container Apps | Consumption | `minReplicas=0`, scales to zero | $0 when idle |
| Azure Container Registry | Basic | 10 GB storage | ~$5 |
| Log Analytics | Pay-as-you-go (PerGB2018) | First 5 GB/month free | ~$0 |
| Key Vault | Standard | Per-operation billing, cents at this scale | <$1 |
| Managed Identity | Free | — | $0 |
| **Total idle** | | | **~$5–$10** |

Active CPU/memory usage during queries adds a few cents per hour of replica
runtime under Consumption pricing — typically negligible for a portfolio
demo. The dominant variable cost is the OpenAI/Anthropic API spend per query
(roughly $0.0001 per query with `gpt-4o-mini`).

---

## 5. Teardown

```bash
az group delete --name rg-guideline-gpt --yes --no-wait
```

That removes every resource the script created.

---

## 6. Troubleshooting

- **`az acr build` fails with an auth error** — run `az login` again; the CLI
  occasionally drops its token.
- **Container starts then 502s** — open the Container App's *Log stream*
  in the portal. Most commonly it's a missing `OPENAI_API_KEY` (which is
  always required, even with `LLM_PROVIDER=anthropic`).
- **First request is slow** — `minReplicas=0` means cold start (~10–20s).
  Increase to `1` in `main.bicep` if you want immediate response.
