# Security Policy

## Supported Versions

Fusion Engine is pre-1.0. Security fixes target the default branch until formal
releases are cut.

## Reporting a Vulnerability

Please report security issues privately to the project maintainers. Do not open a
public issue for secrets, credential exposure, or exploitable service behavior.

## Operating Notes

- Keep `OPENROUTER_API_KEY` out of git. Use environment variables or a local
  `.env`; `.env` is ignored by default.
- Set `FUSION_SERVER_API_KEY` before exposing the HTTP API beyond localhost.
  Requests to `/fuse` and `/v1/chat/completions` must then include
  `Authorization: Bearer <value>`.
- The `code_exec` eval grader executes model-generated Python. Run HumanEval or
  other untrusted code-execution benchmarks inside a container or VM.
