#!/usr/bin/env bash
# Nebius CLI installer + non-interactive service-account auth.
#
# Runs on any machine that submits jobs to Nebius managed compute
# (the CPU app host now; a GPU VM later). Idempotent — safe to re-run.
#
# Auth is non-interactive via a service-account authorized key, so it works
# on headless/remote hosts (no browser OAuth). Required values come from the
# environment (load your .env first), NOT hardcoded:
#
#   NEBIUS_PROFILE               profile name to create (e.g. mlflow-sa)
#   NEBIUS_PROJECT_ID            parent-id the profile operates on
#   NEBIUS_SA_ID                 service account id
#   NEBIUS_SA_PUBLIC_KEY_ID      public key id for that service account
#   NEBIUS_SA_PRIVATE_KEY_FILE   path to the private key .pem on this host
#
# If the auth vars are absent the CLI still installs, and the script prints
# the exact manual command instead of silently skipping.

set -euo pipefail

NEBIUS_ENDPOINT="${NEBIUS_ENDPOINT:-api.eu-north1.nebius.cloud}"
NEBIUS_INSTALL_URL="${NEBIUS_INSTALL_URL:-https://storage.eu-north1.nebius.cloud/cli/install.sh}"
NEBIUS_BIN_DIR="$HOME/.nebius/bin"

echo "=== Nebius CLI install ==="
echo "Endpoint:     $NEBIUS_ENDPOINT"
echo "Install URL:  $NEBIUS_INSTALL_URL"

if ! command -v nebius >/dev/null 2>&1; then
  curl -sSL "$NEBIUS_INSTALL_URL" | bash
else
  echo "nebius already installed at $(command -v nebius)"
fi

# Ensure the CLI is on PATH for this session and future shells.
export PATH="$NEBIUS_BIN_DIR:$PATH"
if ! grep -q '.nebius/bin' "$HOME/.bashrc" 2>/dev/null; then
  echo "export PATH=\"$NEBIUS_BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
  echo "Added $NEBIUS_BIN_DIR to PATH in ~/.bashrc"
fi

echo "=== Nebius CLI version ==="
nebius version

echo "=== Configure service-account profile ==="
: "${NEBIUS_PROFILE:=mlflow-sa}"

# All five auth inputs must be present to create the profile non-interactively.
missing=()
for v in NEBIUS_PROJECT_ID NEBIUS_SA_ID NEBIUS_SA_PUBLIC_KEY_ID NEBIUS_SA_PRIVATE_KEY_FILE; do
  [ -n "${!v:-}" ] || missing+=("$v")
done

if [ "${#missing[@]}" -ne 0 ]; then
  echo "WARNING: missing auth vars: ${missing[*]}"
  echo "CLI is installed but no profile was created. To auth manually, run:"
  echo "  nebius profile create \"\$NEBIUS_PROFILE\" \\"
  echo "    --endpoint $NEBIUS_ENDPOINT \\"
  echo "    --service-account-id <sa-id> \\"
  echo "    --public-key-id <public-key-id> \\"
  echo "    --private-key-file <path.pem> \\"
  echo "    --parent-id <project-id>"
  exit 0
fi

# Fail loudly if the key file is named but not actually present on this host.
if [ ! -f "$NEBIUS_SA_PRIVATE_KEY_FILE" ]; then
  echo "ERROR: NEBIUS_SA_PRIVATE_KEY_FILE=$NEBIUS_SA_PRIVATE_KEY_FILE not found on this host." >&2
  echo "Copy the service-account private key .pem here before running." >&2
  exit 1
fi

echo "Creating profile '$NEBIUS_PROFILE' (parent-id=$NEBIUS_PROJECT_ID, sa=$NEBIUS_SA_ID)"
nebius profile create "$NEBIUS_PROFILE" \
  --endpoint "$NEBIUS_ENDPOINT" \
  --service-account-id "$NEBIUS_SA_ID" \
  --public-key-id "$NEBIUS_SA_PUBLIC_KEY_ID" \
  --private-key-file "$NEBIUS_SA_PRIVATE_KEY_FILE" \
  --parent-id "$NEBIUS_PROJECT_ID"

echo "=== Nebius CLI ready ==="
echo "Active profiles:"
nebius profile list || true
echo "Test with: nebius --profile $NEBIUS_PROFILE iam whoami"
