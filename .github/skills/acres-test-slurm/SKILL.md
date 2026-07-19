---
name: acres-test-slurm
user-invocable: true
description: "Generate ready-to-copy ACRES SLURM commands to run pytest on this repository's files, directories, or test targets."
---

# ACRES SLURM Test Command Generator

This skill produces shell commands or a small `sbatch` submission line for the ACRES supercomputer cluster.

Use when the user wants a copy-paste SLURM command to run tests in this repository, including:
- `python3 -m pytest` targets for single files or directories
- repository-root `cd` and `.venv` activation
- `sbatch`/`srun` flags for a basic test run

## Workflow

1. Identify the test target.
   - If the request names a file path under the repo, use that file path as the pytest target.
   - If the request names a directory, test module, or package, use that as the pytest target.
   - If the request is general, use the repo test command from `README.md`: `python3 -m pytest`.

2. Choose a command style:
   - `sbatch` one-liner for batch submission
   - `srun` interactive execution if the user wants to test quickly on ACRES

3. Build the command with the repo root and virtualenv activation.
   - `cd /home/atotilca/pythongpu`
   - `source .venv/bin/activate`

4. Use `python3 -m pytest <target>` since the repository README and `pytest.ini` show that is the correct test command.
   - Prefer `--maxfail=1 -q` for fast feedback when appropriate.

5. Include a short note that the user may need to adjust ACRES-specific options like `--account`, `--partition`, or `--gres`.

## Output requirements

- Provide commands ready to paste into an ACRES login node shell.
- Use a valid path inside `/home/atotilca/pythongpu`.
- Prefer `sbatch` with `--output=%x-%j.out`, `--time=00:30:00`, and `--mem=8G` unless the user specifies otherwise.
- If a file or directory does not exist in the repo, ask for a valid repo path instead of guessing.

## Example command formats

- Batch submission:
  ```bash
  sbatch --job-name=pythongpu-tests --output=%x-%j.out --time=00:30:00 --mem=8G --wrap='cd /home/atotilca/pythongpu && source .venv/bin/activate && python3 -m pytest tests/test_chimera_classifier.py'
  ```

- Interactive test:
  ```bash
  srun --pty --time=00:30:00 --mem=8G bash
  cd /home/atotilca/pythongpu
  source .venv/bin/activate
  python3 -m pytest tests/test_chimera_classifier.py
  ```

- Whole repository:
  ```bash
  sbatch --job-name=pythongpu-tests --output=%x-%j.out --time=00:30:00 --mem=8G --wrap='cd /home/atotilca/pythongpu && source .venv/bin/activate && python3 -m pytest'
  ```
