FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 1. Setup non-root user and home directory
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /home/app/.streamlit/data /home/app/.streamlit/cache /app

# 2. Set environment variables
ENV \
    WORKDIR=/app \
    HOME=/home/app \
    UV_COMPILE_BYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH="/app/apps/chatbot_ui/src:$PYTHONPATH" \
    PATH="/app/.venv/bin:$PATH"

WORKDIR $WORKDIR

# 3. Copy workspace files and application code
COPY pyproject.toml uv.lock ./
COPY apps/chatbot_ui ./apps/chatbot_ui

# 4. Install dependencies AS ROOT
# We don't switch to 'USER app' yet. This allows us to write to the
# default root cache without needing specific UIDs.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package chatbot_ui

# 5. Fix permissions (The "Option 2" Trade-off)
# Since we installed as root, we must give ownership to 'app' now.
# This ensures the app user can read the venv and run the code.
RUN chown -R app:app /app /home/app

# 6. Runtime configuration
EXPOSE 8501

# Finally, switch to the non-root user for runtime security
USER app

WORKDIR /app/apps/chatbot_ui/src

CMD ["uv", "run", "streamlit", "run", "chatbot_ui/app.py", "--server.address=0.0.0.0"]