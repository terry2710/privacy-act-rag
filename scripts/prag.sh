#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  . "$ROOT/.venv/bin/activate"
fi

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:$PORT}"
OFFLINE_MARKERS="not aws and not index"

usage() {
  cat <<'USAGE'
usage: prag <command> [args]

quality
  test          offline pytest suite (same selection as CI)
  api-test      tests/test_api.py only
  gate          retrieval regression gate (check_eval_regression.py)
  check         compile check + offline tests + regression gate

run
  api           uvicorn api.main:app with reload (PORT, default 8000)
  gradio        python demos/gradio/app.py
  streamlit     streamlit run demos/streamlit/app.py (needs demos/streamlit/requirements.txt)

call a running API (BASE_URL, default http://localhost:$PORT)
  live          GET /health/live
  ready         GET /health/ready with status code
  ask <text>    POST /api/queries and pretty-print the response

stack
  up            start Postgres and the local queue
  down          stop all compose services
  stack         build and start every compose service
  logs          follow compose logs
  migrate       alembic upgrade head
  image         build the application image
  web           start the front-end dev server
  web-check     front-end lint, type check, tests and build

cluster
  kind-up       create the kind cluster
  kind-load     load the application image into kind
  kind-down     delete the kind cluster
  k-secret      create the database secret in the cluster
  k-apply       apply the manifests in infra/k8s
  tf-check      terraform fmt, init and validate
  helm-lint     helm lint the chart

git helpers (read-only except commit)
  status        git status -sb
  log           last 10 commits
  msgs          list files in .commit_msgs
  commit <file> git commit -F .commit_msgs/<file> (stage first with git add)
USAGE
}

compile_all() {
  python -m py_compile core/*.py api/*.py worker/*.py tools/*.py demos/gradio/app.py demos/streamlit/app.py
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  test)
    python -m pytest -m "$OFFLINE_MARKERS" "$@"
    ;;
  api-test)
    python -m pytest tests/test_api.py "$@"
    ;;
  gate)
    python -m tools.check_eval_regression "$@"
    ;;
  check)
    compile_all
    python -m pytest -m "$OFFLINE_MARKERS"
    python -m tools.check_eval_regression
    ;;
  api)
    exec python -m uvicorn api.main:app --reload --port "$PORT" "$@"
    ;;
  gradio)
    exec python demos/gradio/app.py "$@"
    ;;
  streamlit)
    exec python -m streamlit run demos/streamlit/app.py "$@"
    ;;
  live)
    curl -fsS "$BASE_URL/health/live"
    echo
    ;;
  ready)
    curl -sS -w '\nHTTP %{http_code}\n' "$BASE_URL/health/ready"
    ;;
  ask)
    if [ "$#" -eq 0 ]; then
      echo "usage: prag ask <question>" >&2
      exit 1
    fi
    body="$(python -c 'import json,sys; print(json.dumps({"question": " ".join(sys.argv[1:])}))' "$@")"
    curl -sS -X POST "$BASE_URL/api/queries" -H "content-type: application/json" -d "$body" | python -m json.tool
    ;;
  up)
    docker compose up -d postgres elasticmq "$@"
    ;;
  down)
    docker compose down "$@"
    ;;
  stack)
    docker compose up --build "$@"
    ;;
  logs)
    docker compose logs -f "$@"
    ;;
  migrate)
    python -m alembic -c migrations/alembic.ini upgrade head
    ;;
  image)
    docker build -t privacy-act-rag:dev .
    ;;
  web)
    cd frontend
    npm run dev
    ;;
  web-check)
    cd frontend
    npm run lint
    npx tsc -b
    npm test
    npm run build
    ;;
  kind-up)
    kind create cluster --config infra/kind/kind-config.yaml
    ;;
  kind-load)
    kind load docker-image privacy-act-rag:dev --name prag
    ;;
  kind-down)
    kind delete cluster --name prag
    ;;
  k-secret)
    kubectl -n privacy-act-rag create secret generic api-secrets --from-literal=database-url=postgresql+psycopg://app:app@postgres:5432/privacyrag --dry-run=client -o yaml | kubectl apply -f -
    ;;
  k-apply)
    kubectl apply -f infra/k8s/namespace.yaml
    kubectl apply -f infra/k8s/
    ;;
  tf-check)
    terraform -chdir=infra/terraform fmt -check -recursive
    terraform -chdir=infra/terraform init -backend=false
    terraform -chdir=infra/terraform validate
    ;;
  helm-lint)
    helm lint infra/helm/privacy-act-rag
    ;;
  status)
    git status -sb
    ;;
  log)
    git log --oneline -10
    ;;
  msgs)
    ls -1 .commit_msgs
    ;;
  commit)
    name="${1:-}"
    if [ -z "$name" ] || [ ! -f ".commit_msgs/$name" ]; then
      echo "usage: prag commit <file in .commit_msgs>" >&2
      ls -1 .commit_msgs 2>/dev/null >&2 || true
      exit 1
    fi
    git commit -F ".commit_msgs/$name"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
