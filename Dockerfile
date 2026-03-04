# Multi-stage build for minimal production image
# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir build \
    && python -m build --wheel \
    && pip install --no-cache-dir --prefix=/install dist/*.whl


# Stage 2: Production
FROM python:3.11-slim AS production

# Security: run as non-root user
RUN groupadd --gid 1000 ganglion \
    && useradd --uid 1000 --gid ganglion --shell /bin/sh --create-home ganglion

COPY --from=builder /install /usr/local

# Ensure the package is importable
RUN python -c "import ganglion"

WORKDIR /app

USER ganglion

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/healthz')" || exit 1

EXPOSE 8899

# Log to stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV GANGLION_HOST=0.0.0.0
ENV GANGLION_PORT=8899

ENTRYPOINT ["ganglion"]
CMD ["serve", "/app/project", "--host", "0.0.0.0", "--port", "8899"]
