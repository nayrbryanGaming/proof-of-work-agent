# Self-hosted Runner — setup guide

This document explains how to set up a GitHub Actions self-hosted runner for the `proof-of-work-agent` repository so your workflows can run on a machine you control (24/7).

Requirements
- A Linux machine or VM (Ubuntu recommended) or Windows (see notes).
- A user with sudo privileges to install the runner/service.

Quick steps (Linux)
1. On GitHub: go to `Settings -> Actions -> Runners -> New self-hosted runner` for the repository and copy the registration token.
2. SSH into your server and run (replace TOKEN):
```bash
git clone https://github.com/nayrbryanGaming/proof-of-work-agent.git
cd proof-of-work-agent/scripts
./setup_selfhosted_runner.sh YOUR_REGISTRATION_TOKEN "my-runner-name"
```
3. The script will download the runner, configure it, install and start it as a service. The runner will appear in the repository settings.

Change workflows to use the runner
1. Edit the workflow `runs-on:` to include `self-hosted` and any labels you set, for example:
```yaml
runs-on: [self-hosted, linux]
```

Windows notes
- Download the runner for Windows from the GitHub UI during runner setup.
- Run `config.cmd` and use `run.cmd` or install as service using the provided `svc.sh` equivalent or NSSM.

Security & maintenance
- Keep the machine patched and limit network access.
- Use an isolated service account for the runner.
- Rotate registration tokens if the machine is compromised.

If you want, I can also add a `systemd` unit template, or a Windows service helper for NSSM.
