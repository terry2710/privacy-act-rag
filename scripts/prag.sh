#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f "$ROOT/.venv/bin/activate" ]; then
  . "$ROOT/.venv/bin/activate"
fi

if [ -f "$ROOT/.env" ]; then
  set -a
  . "$ROOT/.env"
  set +a
fi

PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://localhost:$PORT}"
OFFLINE_MARKERS="not aws and not index"
K8S_NAMESPACE="privacy-act-rag"
KIND_CLUSTER="prag"
IMAGE_TAG="$(git describe --always --dirty 2>/dev/null || echo dev)"

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
  build-index   load the index from cache or S3, or build it (costs Bedrock calls); --check never builds

call a running API (BASE_URL, default http://localhost:$PORT)
  live          GET /health/live
  ready         GET /health/ready with status code
  ask <text>    POST /api/queries and pretty-print the response

stack
  up            start Postgres and the local queue
  down          stop all compose services
  stack         build and start every compose service
  stack-migrate run database migrations inside the api container
  logs          follow compose logs
  migrate       alembic upgrade head
  db-check      alembic check: models and database schema agree
  image         build the application image
  web           start the front-end dev server
  web-check     front-end lint, type check, tests and build

cluster
  kind-up       create the kind cluster
  kind-down     delete the kind cluster
  k-secrets     create the namespace, database secret and AWS credentials secret
  k-deploy      build, load into kind and roll out the image tagged with the git revision
  k-migrate     run the database migration Job with the current image
  k-status      list pods, deployments, services and jobs
  k-logs        follow the api logs
  k-forward     port-forward the api service to localhost:PORT
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
  build-index)
    python -m tools.build_index "$@"
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
    APP_VERSION="$IMAGE_TAG" docker compose up --build "$@"
    ;;
  stack-migrate)
    docker compose run --rm api python -m alembic -c migrations/alembic.ini upgrade head
    ;;
  logs)
    docker compose logs -f "$@"
    ;;
  migrate)
    python -m alembic -c migrations/alembic.ini upgrade head
    ;;
  db-check)
    python -m alembic -c migrations/alembic.ini check
    ;;
  image)
    docker build --build-arg APP_VERSION="$IMAGE_TAG" -t privacy-act-rag:dev -t "privacy-act-rag:$IMAGE_TAG" .
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
    kubectl cluster-info --context "kind-$KIND_CLUSTER"
    ;;
  kind-down)
    kind delete cluster --name "$KIND_CLUSTER"
    ;;
  k-secrets)
    kubectl create namespace "$K8S_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    kubectl -n "$K8S_NAMESPACE" create secret generic api-secrets --from-literal=database-url=postgresql+psycopg://app:app@postgres:5432/privacyrag --dry-run=client -o yaml | kubectl apply -f -
    kubectl -n "$K8S_NAMESPACE" create secret generic aws-credentials --from-file=credentials="$HOME/.aws/credentials" --dry-run=client -o yaml | kubectl apply -f -
    ;;
  k-deploy)
    docker build --build-arg APP_VERSION="$IMAGE_TAG" -t privacy-act-rag:dev -t "privacy-act-rag:$IMAGE_TAG" .
    archive="$(mktemp -t prag-image)"
    docker save -o "$archive" "privacy-act-rag:$IMAGE_TAG"
    kind load image-archive "$archive" --name "$KIND_CLUSTER"
    rm -f "$archive"
    kubectl kustomize infra/k8s | sed "s#privacy-act-rag:dev#privacy-act-rag:$IMAGE_TAG#" | kubectl apply -f -
    kubectl -n "$K8S_NAMESPACE" rollout status deployment/api --timeout=180s
    ;;
  k-migrate)
    kubectl -n "$K8S_NAMESPACE" delete job migrate --ignore-not-found
    sed "s#privacy-act-rag:dev#privacy-act-rag:$IMAGE_TAG#" infra/k8s/jobs/migrate.yaml | kubectl apply -f -
    kubectl -n "$K8S_NAMESPACE" wait --for=condition=complete job/migrate --timeout=180s
    kubectl -n "$K8S_NAMESPACE" logs job/migrate
    ;;
  k-status)
    kubectl -n "$K8S_NAMESPACE" get pods,deployments,services,jobs
    ;;
  k-logs)
    kubectl -n "$K8S_NAMESPACE" logs deployment/api -f
    ;;
  k-forward)
    kubectl -n "$K8S_NAMESPACE" port-forward service/api "$PORT:8000"
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
