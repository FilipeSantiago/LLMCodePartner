// Text-only front end for context-compress's output pipeline.
//
// Code Partner sends text it already holds and authorized; this process only
// transforms it. It never receives a command, never runs one, and never reads
// project files. The upstream `execute`/`fetch_and_index` tools and the `auto`
// mode (which calls an LLM) are unreachable from here.
//
// Protocol: newline-delimited JSON, one request per line, one response per line.
//   {"id": 1, "op": "compress", "mode": "conservative|balanced", "text": "..."}
//   {"id": 2, "op": "version"}
// Transports: `--stdio` (local backend) or `--socket <path>` (sidecar).
// `--health <path>` probes a running sidecar and exits 0 when it answers.

import { createRequire, syncBuiltinESMExports } from "node:module";
import net from "node:net";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";

const MODES = new Set(["conservative", "balanced"]);
const MAX_LINE_BYTES = 16 * 1024 * 1024;

// The filter path makes no network calls; make sure a future upstream change
// cannot either. Process spawning and file writes are already denied by
// Node's permission model (see Dockerfile / local command).
const denied = () => {
	throw new Error("network access is disabled in compress-bridge");
};
globalThis.fetch = denied;
for (const name of ["connect", "createConnection"]) {
	if (process.argv[2] !== "--health") net[name] = denied;
}
syncBuiltinESMExports();

const require = createRequire(import.meta.url);
const pkgJson = require.resolve("context-compress/package.json");
const { version: VERSION } = require(pkgJson);
const { compressOutput } = await import(join(dirname(pkgJson), "dist", "cli", "filter.js"));

export function handle(request) {
	if (request === null || typeof request !== "object") return { ok: false, error: "request must be an object" };
	const { id = null, op } = request;
	if (op === "version") return { id, ok: true, version: VERSION };
	if (op !== "compress") return { id, ok: false, error: `unsupported op ${JSON.stringify(op)}` };
	if ("cmd" in request || "command" in request || "hint" in request) {
		return { id, ok: false, error: "command hints are not accepted" };
	}
	if (!MODES.has(request.mode)) return { id, ok: false, error: `mode ${JSON.stringify(request.mode)} is not allowed` };
	if (typeof request.text !== "string") return { id, ok: false, error: "text must be a string" };
	// An explicit cap at the input size disables upstream smartTruncate: bounding
	// is Code Partner's decision, and silent truncation is never acceptable.
	const cap = Math.max(1, Buffer.byteLength(request.text) + 1);
	const text = compressOutput(request.text, undefined, request.mode, cap);
	return { id, ok: true, version: VERSION, text };
}

function serve(input, write) {
	const lines = createInterface({ input, crlfDelay: Number.POSITIVE_INFINITY });
	lines.on("line", (line) => {
		let response;
		try {
			response = line.length > MAX_LINE_BYTES
				? { ok: false, error: "request too large" }
				: handle(JSON.parse(line));
		} catch (error) {
			response = { ok: false, error: String(error?.message ?? error) };
		}
		write(`${JSON.stringify(response)}\n`);
	});
	return lines;
}

const [flag, value] = process.argv.slice(2);
if (flag === "--stdio") {
	serve(process.stdin, (text) => process.stdout.write(text));
} else if (flag === "--socket" && value) {
	const server = net.createServer((socket) => {
		socket.setEncoding("utf8");
		serve(socket, (text) => socket.write(text));
		socket.on("error", () => socket.destroy());
	});
	process.umask(0o007);
	server.listen(value, () => process.stderr.write(`compress-bridge ${VERSION} listening on ${value}\n`));
	for (const signal of ["SIGTERM", "SIGINT"]) process.on(signal, () => server.close(() => process.exit(0)));
} else if (flag === "--health" && value) {
	const socket = net.createConnection(value);
	socket.setEncoding("utf8");
	socket.setTimeout(3000, () => process.exit(1));
	socket.on("error", () => process.exit(1));
	socket.on("connect", () => socket.write(`${JSON.stringify({ id: 0, op: "version" })}\n`));
	socket.on("data", (chunk) => process.exit(chunk.includes('"ok":true') ? 0 : 1));
} else {
	process.stderr.write("usage: bridge.mjs --stdio | --socket <path> | --health <path>\n");
	process.exit(2);
}
