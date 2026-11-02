# Offline test image: Python (stdlib core + the optional MCP SDK), Node + Pi (for the bridge tests against Pi's
# offline faux provider) and ffmpeg (eval-judge sheet tests). No model calls, no credentials.
#   docker build -f tests/docker/Dockerfile -t train-snitch-test .
#   docker run --rm --network none train-snitch-test
FROM node:24-bookworm-slim
ARG PI_VERSION=0.80.10
ARG MCP_SPEC="mcp>=2.0,<3"
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg procps tzdata \
    && rm -rf /var/lib/apt/lists/*
RUN npm install -g "@earendil-works/pi-coding-agent@${PI_VERSION}" && npm cache clean --force
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir "${MCP_SPEC}" "setuptools>=70" wheel
ENV PATH=/opt/venv/bin:$PATH PI_OFFLINE=1
WORKDIR /src
COPY . /src
RUN useradd -m tester && chown -R tester /src
USER tester
CMD ["python3", "-I", "-m", "unittest", "discover", "-s", "tests", "-v"]
