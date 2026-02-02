FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Create non-root user and home directory first
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    mkdir -p /home/app /app

# Set environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PYTHONOPTIMIZE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/app
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/apps/api/src:$PYTHONPATH"

WORKDIR /app

# Copy workspace root files for dependency resolution
COPY pyproject.toml uv.lock ./

# Copy application code (needed before uv sync for workspace packages)
COPY apps/api ./apps/api

# Install dependencies AS ROOT (before switching to app user)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api

# Fix permissions - give ownership to app user
RUN chown -R app:app /app /home/app

# Switch to non-root user
USER app

# Expose the FastAPI port
EXPOSE 8000

WORKDIR /app/apps/api/src

# Command to run the application
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]