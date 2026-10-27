# Security policy

## Supported versions

Security fixes are applied to the latest released GeoUG version. Research
snapshots and unreleased checkpoints are not supported.

## Reporting a vulnerability

Do not open a public issue for credentials, arbitrary code execution,
dependency compromise, unsafe model deserialization, or private-data
exposure. Use **Security → Report a vulnerability** in the public GitHub
repository; maintainers must enable GitHub private vulnerability reporting
before creating the first release tag. Include:

- the affected commit and file;
- reproduction steps and required environment;
- expected impact;
- whether any credential or private data may have been exposed.

## Model and dataset safety

GeoUG model outputs and generated geometry are predictions, not calibrated
measurements. Do not use them for safety-critical navigation or control.
Report suspected benchmark image-license or privacy issues through the same
private channel.
