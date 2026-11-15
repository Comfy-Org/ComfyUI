# Security Policy

## Scope

ComfyUI is designed to run locally. By default, the server binds to `127.0.0.1`, meaning only the user's own machine can reach it. Our threat model assumes:

- The user installed ComfyUI through a supported channel: the desktop application, the portable build, or a manual install following the README.
- The user has not installed untrusted custom nodes. Custom nodes are arbitrary Python code and are trusted as much as any other software the user chooses to install.
- Anyone with access to the ComfyUI URL is trusted (a direct consequence of the localhost-only default).
- PyTorch and other dependencies are at the versions we ship or recommend in the README.

A report is in scope only if it affects a user operating within this threat model.

## What We Consider a Vulnerability

We want to hear about issues where a **reasonable user** — someone who does not install random untrusted nodes and who reads UI prompts and warnings before clicking through them — can be harmed by ComfyUI itself.

The clearest example: a workflow file that such a user might plausibly load and run, using only built-in nodes, that results in **untrusted code execution, arbitrary file read/write outside expected directories, or credential/data exfiltration**.

When submitting a report, please include a clear description of *why this is a problem for a typical local ComfyUI user*. Reports without this context are difficult to act on.

## What We Do Not Consider a Security Vulnerability

Please report the following through our regular [GitHub issues](https://github.com/comfyanonymous/ComfyUI/issues) instead. Filing them as security reports will likely cause them to be deprioritized or closed.

- **Issues requiring `--listen` or any non-default network exposure.** ComfyUI binds to localhost by default. If a remote attacker needs to reach the server for the attack to work, the user has chosen to expose it and is responsible for securing that deployment (firewall, reverse proxy, authentication, etc.). These are bugs, not vulnerabilities.
- **`torch.load` and related deserialization issues in old PyTorch versions.** These are upstream PyTorch issues. Our distributions ship with — and our documentation recommends — recent PyTorch versions where these are addressed.
- **Vulnerabilities that depend on outdated library versions** that we neither ship nor recommend (e.g., requiring PyTorch 2.6 or older).
- **Issues that require a specific custom node to be installed.** Custom nodes are third-party code. Report these to the maintainer of that node.
- **Crashes, hangs, or resource exhaustion from a loaded workflow.** Annoying, but not a security issue in our model. File a regular bug.
- **Social-engineering scenarios** where the user is expected to ignore an explicit UI warning or prompt.

## Reporting

If you believe you have found an issue that falls within the scope above, please report it privately via GitHub's [Report a vulnerability](https://github.com/comfyanonymous/ComfyUI/security/advisories/new) feature rather than opening a public issue.

Please include:

1. A description of the vulnerability and the affected component.
2. Reproduction steps, ideally with a minimal workflow file or proof-of-concept.
3. The ComfyUI version, install method (desktop / portable / manual), and OS.
4. An explanation of how this affects a typical local user as described in the threat model.

We will acknowledge valid reports and coordinate a fix and disclosure timeline with you.
