// guideline-gpt — Azure Container Apps deployment.
//
// Provisions:
//   - Log Analytics workspace (for container logs)
//   - Container Apps managed environment, bound to the workspace
//   - User-assigned managed identity (ACR pull + Key Vault read)
//   - Key Vault with RBAC, holding the LLM/embedding API keys
//   - Container App referencing the image in an EXISTING ACR
//
// The ACR is pre-created by deploy.sh so the image can be built and pushed
// before the Container App is provisioned.

@description('Application name; used as the Container App name and resource prefix.')
param appName string = 'guideline-gpt'

@description('Azure region for all resources.')
param location string = resourceGroup().location

@description('Name of an existing Azure Container Registry holding the image.')
param acrName string

@description('Image tag pushed to ACR (e.g. "latest" or a git SHA).')
param imageTag string = 'latest'

@description('LLM provider for the app to use.')
@allowed([
  'anthropic'
  'openai'
])
param llmProvider string = 'anthropic'

@description('Anthropic model name passed to the app.')
param anthropicModel string = 'claude-haiku-4-5-20251001'

@description('OpenAI model name passed to the app.')
param openaiModel string = 'gpt-4o-mini'

@secure()
@description('OpenAI API key (ALWAYS required: powers embeddings).')
param openaiApiKey string

@secure()
@description('Anthropic API key. Required only when llmProvider=anthropic; pass an empty string otherwise.')
param anthropicApiKey string = ''

// --- naming -----------------------------------------------------------------

var sanitized = toLower(replace(appName, '-', ''))
var lawName = '${appName}-law'
var envName = '${appName}-env'
var identityName = '${appName}-identity'
// Key Vault names are global and limited to 24 chars.
var kvName = take('${sanitized}kv${uniqueString(resourceGroup().id)}', 24)
var containerAppName = appName

// --- references -------------------------------------------------------------

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' existing = {
  name: acrName
}

// --- observability ----------------------------------------------------------

resource law 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: lawName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource managedEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: envName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: law.properties.customerId
        sharedKey: law.listKeys().primarySharedKey
      }
    }
  }
}

// --- identity + RBAC --------------------------------------------------------

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// Built-in role IDs (stable across subscriptions).
var acrPullRoleId = '7f951dda-4ed3-11e8-a3ec-22000a006b8c'
var kvSecretsUserRoleId = '4633458b-17de-408a-b874-0445c86b69e6'

resource acrPullAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, identity.id, acrPullRoleId)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      acrPullRoleId
    )
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// --- secrets ----------------------------------------------------------------

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: kvName
  location: location
  properties: {
    enableRbacAuthorization: true
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: false
    tenantId: tenant().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    accessPolicies: []
  }
}

resource kvAccessAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(kv.id, identity.id, kvSecretsUserRoleId)
  scope: kv
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      kvSecretsUserRoleId
    )
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

resource openaiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: kv
  name: 'openai-api-key'
  properties: {
    value: openaiApiKey
  }
}

resource anthropicKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = if (!empty(anthropicApiKey)) {
  parent: kv
  name: 'anthropic-api-key'
  properties: {
    value: anthropicApiKey
  }
}

// --- container app ----------------------------------------------------------

var baseSecrets = [
  {
    name: 'openai-api-key'
    identity: identity.id
    keyVaultUrl: '${kv.properties.vaultUri}secrets/openai-api-key'
  }
]
var anthropicSecretEntry = !empty(anthropicApiKey)
  ? [
      {
        name: 'anthropic-api-key'
        identity: identity.id
        keyVaultUrl: '${kv.properties.vaultUri}secrets/anthropic-api-key'
      }
    ]
  : []

var baseEnv = [
  {
    name: 'LLM_PROVIDER'
    value: llmProvider
  }
  {
    name: 'ANTHROPIC_MODEL'
    value: anthropicModel
  }
  {
    name: 'OPENAI_MODEL'
    value: openaiModel
  }
  {
    name: 'OPENAI_API_KEY'
    secretRef: 'openai-api-key'
  }
]
var anthropicEnvEntry = !empty(anthropicApiKey)
  ? [
      {
        name: 'ANTHROPIC_API_KEY'
        secretRef: 'anthropic-api-key'
      }
    ]
  : []

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  dependsOn: [
    acrPullAssignment
    kvAccessAssignment
    openaiKeySecret
  ]
  properties: {
    environmentId: managedEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: identity.id
        }
      ]
      secrets: concat(baseSecrets, anthropicSecretEntry)
    }
    template: {
      containers: [
        {
          name: 'app'
          image: '${acr.properties.loginServer}/${appName}:${imageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: concat(baseEnv, anthropicEnvEntry)
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
      }
    }
  }
}

// --- outputs ----------------------------------------------------------------

output appFqdn string = containerApp.properties.configuration.ingress.fqdn
output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output keyVaultName string = kv.name
output managedIdentityClientId string = identity.properties.clientId
