# Contributing to GeoUG

Thank you for improving GeoUG. Please keep changes focused on reproducibility,
portability, or correctness of the released method.

## Before opening a change

1. Do not add model weights, datasets, generated outputs, credentials, or
   machine-specific paths.
2. Preserve third-party copyright and SPDX headers.
3. Add or update a CPU test for behavior that does not require a GPU.
4. Run `bash scripts/check_release.sh`.
5. For GPU changes, record the checkpoint, command, hardware, seed, and minimal
   output needed to reproduce the result.

## Code and data changes

- Public paths must be command-line arguments, configuration values, or
  environment variables.
- New benchmark data must document its source, schema, coordinate convention,
  and checksum.
- Metric changes require a regression fixture with an independently known
  answer.
- Evaluation prompts are part of the benchmark protocol. Prompt changes must
  include captured-prompt tests for every affected model family.

## Licensing

GeoUG is mixed-license. New first-party files should use:

```text
Copyright 2026 ByteDance Ltd. and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
```

Do not copy external code without retaining its license and adding it to
`THIRD_PARTY_NOTICES.md`.
