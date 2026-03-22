# Project Plan

This file is the top-level reference for the numerical claim detection project.

## Which document to follow

There are two planning documents in this project, and they do not serve the same purpose.

### 1. Teacher-facing study plan

File:

- `plans/study_plan_numerical_claims.tex`
- `plans/study_plan_numerical_claims.pdf`

Purpose:

- this is the clean version intended for review, discussion, or submission
- it presents the project in a concise academic style
- it is intentionally simpler and less operational

This document explains the project clearly, but it should **not** be treated as the detailed implementation specification.

### 2. Internal implementation note

File:

- `plans/PROJECT_DIRECTION.md`

Purpose:

- this is the document we follow when building the project
- it contains the exact two-stage structure
- it records the restricted Stage 2 design
- it contains the operational decisions that are too detailed for the study plan

This document **is** the implementation reference.

## Current rule

For implementation:

- follow `plans/PROJECT_DIRECTION.md`

For communication with the teacher:

- use `plans/study_plan_numerical_claims.tex` / `plans/study_plan_numerical_claims.pdf`

## Why this split exists

The study plan is meant to be readable, defensible, and appropriately scoped for a course discussion. The implementation note is meant to keep the project precise and prevent scope drift once coding starts.

So the practical rule is simple:

- the study plan explains the project
- the project direction note defines what we actually build
