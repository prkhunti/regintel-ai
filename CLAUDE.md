# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project

**regintel-ai** is a portfolio / reference implementation of audit-ready clinical document intelligence for regulated AI workflows.

The repository contains a FastAPI API, Celery ingestion worker, Next.js web app, shared retrieval/evaluation packages, Docker Compose infrastructure, and pytest coverage.

## Working Guidelines

- Keep changes focused and consistent with the existing architecture.
- Do not add fake metrics, screenshots, benchmarks, usage numbers, or deployment claims.
- Do not commit secrets, credentials, private endpoints, protected health information, or private datasets.
- Prefer documentation and setup polish over broad refactors unless a code change is clearly needed to fix setup.
- Treat this as a showcase-first repository, not a contribution-driven open source project.
