FROM node:22.13.0-bookworm-slim

WORKDIR /opt/compress-bridge
COPY infra/compress-bridge/package.json infra/compress-bridge/package-lock.json ./
# Install scripts would build better-sqlite3 for the upstream MCP server, which
# this bridge never loads; the filter pipeline is plain JavaScript.
RUN npm ci --ignore-scripts --omit=dev --no-audit --no-fund && npm cache clean --force
COPY infra/compress-bridge/bridge.mjs ./bridge.mjs

# Same uid/gid as the codepartner user so only the two containers share the socket.
RUN mkdir -p /run/compress-bridge && chown 10001:10001 /run/compress-bridge
USER 10001:10001

# Node's permission model: read only the bridge itself; write only the socket
# directory; no child processes, workers, native addons or WASI.
CMD ["node", "--permission", "--allow-fs-read=/opt/compress-bridge", \
     "--allow-fs-write=/run/compress-bridge", \
     "/opt/compress-bridge/bridge.mjs", "--socket", "/run/compress-bridge/compress.sock"]
