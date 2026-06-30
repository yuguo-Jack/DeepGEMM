---
name: tx32-supernode-workflow
description: Operate the TX32 two-node Hygon DCU supernode for DeepGEMM/MegaMoE. Use when Codex needs to sync the local workspace to both non-shared-storage nodes, SSH to root@10.17.160.69 and root@10.17.162.22, run compile/test/debug commands inside docker container yiqa_deepep, verify /home/yiqa host-to-container mounts, inspect DCU state, or launch 32-card jobs with torchrun multi-node IP/rendezvous across two 16-card nodes.
---

# TX32 Supernode Workflow

## Execution Contract

- Edit code only in the local repository.
- The two TX32 nodes do not share storage. Any required source, config, script, or test file must be synced to both nodes.
- Keep the same path on both nodes: host `/home/yiqa/DeepGEMM`, container `/home/yiqa/DeepGEMM`.
- Run project work inside Docker: `docker exec yiqa_deepep bash -lc 'source /home/yiqa/dtk-26.04.1/env.sh 2>/dev/null || source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && <cmd>'`.
- Host-side Docker inspection commands such as `docker ps`, `docker inspect`, and `docker start` may run outside Docker.
- Avoid direct host-side compilation/testing outside Docker unless the user explicitly requests it.
- Because SSH login is `root`, ask the user before host-side deletion, recursive move, `chmod`, `chown`, package install, service restart, driver/network edits, or cleanup outside `/home/yiqa/DeepGEMM`.
- Creating `/home/yiqa/DeepGEMM` and syncing local project files into that path is allowed when the user asks for remote setup/sync.
- Put all remote temporary files, logs, profiler output, and debug artifacts under `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/`.
- Do not modify host system files. Do not touch Docker daemon, OS packages, driver files, SSH server config, or network config.

## Fixed Topology

- Node 0: `root@10.17.160.69`
- Node 1: `root@10.17.162.22`
- SSH key: `C:/Users/Administrator/.ssh/id_rsa`
- SSH port: `22`
- Docker name on both nodes: `yiqa_deepep`
- Host/container mount on both nodes: `/home/yiqa` -> `/home/yiqa`
- Container repo path on both nodes: `/home/yiqa/DeepGEMM`
- Preferred DTK environment on both nodes: `/home/yiqa/dtk-26.04.1/env.sh`; use `/opt/dtk/env.sh` only as a fallback when that path is unavailable.
- Each node has 16 local DCUs. Treat the 32-card supernode as two 16-card nodes, not as one host with local devices `0..31`.
- Select the launch model from the target program, not from card count alone:
  - SGLang / DeepEP / MegaMoE local-spawn tests usually use outer `torchrun --nproc-per-node=1`: one launcher process per node, then the script's `--num-processes=16` spawns one local worker per DCU inside that node.
  - Training stacks such as Megatron or other standard one-process-per-device DDP programs usually use `torchrun --nproc-per-node=16`, because torchrun itself owns the per-DCU worker creation.
  - Do not combine `torchrun --nproc-per-node=16` with a script that also does `torch.multiprocessing.spawn(..., nprocs=16)`, or it will oversubscribe each node.

## Read Parameters

Read `.vscode/tx_supernode32.json` first. It should be a JSON array with two entries. Each entry should include:

```json
{
  "name": "supernode32-node69",
  "context": ".",
  "host": "10.17.160.69",
  "username": "root",
  "privateKeyPath": "C:/Users/Administrator/.ssh/id_rsa",
  "remotePath": "/home/yiqa/DeepGEMM"
}
```

`context: "."` means the local VS Code workspace root. Both entries should point to the same local context and the same remote path because both nodes need identical files.

## Windows SSH Reliability

- Prefer `ssh -F NUL ...` and `scp -F NUL ...` to avoid local SSH config surprises.
- Use `-o BatchMode=yes` for checks that must prove key-based auth.
- Use `-o StrictHostKeyChecking=accept-new` for first contact with the known TX32 nodes.
- If the private key has Windows ACL issues, copy it to a temporary file with restricted ACLs before retrying.

## Quick Verification Workflow

Run these checks on both nodes before build/test/launch:

```powershell
ssh -F NUL -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "hostname && whoami && test -d /home/yiqa && echo host_home_ok=/home/yiqa"
ssh -F NUL -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "hostname && whoami && test -d /home/yiqa && echo host_home_ok=/home/yiqa"
```

Check Docker and mount:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker ps -a --filter name=yiqa_deepep --format 'table {{.Names}}\t{{.Status}}' && docker inspect yiqa_deepep --format '{{json .Mounts}}'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker ps -a --filter name=yiqa_deepep --format 'table {{.Names}}\t{{.Status}}' && docker inspect yiqa_deepep --format '{{json .Mounts}}'"
```

Check repo visibility inside Docker after sync:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; test -d /home/yiqa/DeepGEMM && cd /home/yiqa/DeepGEMM && echo container_repo_ok=/home/yiqa/DeepGEMM && pwd && (which hipcc || true) && python3 -V && pip3 --version'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; test -d /home/yiqa/DeepGEMM && cd /home/yiqa/DeepGEMM && echo container_repo_ok=/home/yiqa/DeepGEMM && pwd && (which hipcc || true) && python3 -V && pip3 --version'"
```

## Sync Workflow

Treat sync as a two-way workflow:

- Local workspace is the primary editing location.
- Remote nodes are the primary execution locations.
- The two nodes do not share storage, so every local-to-remote sync must update both nodes.
- Keep the local workspace, node0 `/home/yiqa/DeepGEMM`, and node1 `/home/yiqa/DeepGEMM` synchronized for all source/config/test files needed by a run.
- Remote-generated logs, profiles, and debug outputs under `hygon_tmp/supernode_debug/` may be copied back to local for analysis.
- Do not rely on remote Git state. The remote checkout may be a plain copied workspace, and Git commands there are optional diagnostics only.

Verify the repo path exists on both nodes:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "test -d /home/yiqa/DeepGEMM && echo node0_repo_ok=/home/yiqa/DeepGEMM"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "test -d /home/yiqa/DeepGEMM && echo node1_repo_ok=/home/yiqa/DeepGEMM"
```

Only create `/home/yiqa/DeepGEMM` during first-time node setup if it is missing and the user asked for setup/sync.

For normal editing, prefer the VS Code SFTP multi-context config `.vscode/tx_supernode32.json` so saves upload to both nodes. If upload-on-save is uncertain, explicitly upload the touched files or directories to both nodes with `scp` or SFTP.

Explicit single-file upload pattern:

```powershell
scp -F NUL -P 22 -i C:/Users/Administrator/.ssh/id_rsa <local_file> "root@10.17.160.69:/home/yiqa/DeepGEMM/<relative_path>"
scp -F NUL -P 22 -i C:/Users/Administrator/.ssh/id_rsa <local_file> "root@10.17.162.22:/home/yiqa/DeepGEMM/<relative_path>"
```

Use full-workspace sync when setting up a fresh node or when many files changed. Use targeted sync for iterative edits. Do not delete stale remote files unless the user explicitly approves the cleanup and the delete scope is restricted to `/home/yiqa/DeepGEMM`.

Temporary file policy:

- Put local temporary files under the local repo's `hygon_tmp/supernode_debug/` or remove them after use.
- Put all remote execution logs, profiles, and debug artifacts under `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/`.
- Remote temporary/debug artifacts do not all need to be copied back to local. Prefer keeping corresponding run directories organized and comparable on both remote nodes.
- Copy selected remote artifacts back from each node only when local analysis needs them:

```powershell
scp -F NUL -P 22 -i C:/Users/Administrator/.ssh/id_rsa -r "root@10.17.160.69:/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/<run>" "hygon_tmp/supernode_debug/tx32_node0_<run>"
scp -F NUL -P 22 -i C:/Users/Administrator/.ssh/id_rsa -r "root@10.17.162.22:/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/<run>" "hygon_tmp/supernode_debug/tx32_node1_<run>"
```

After sync, verify both host and container paths see the expected file:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "test -f /home/yiqa/DeepGEMM/<relative_path> && docker exec yiqa_deepep bash -lc 'test -f /home/yiqa/DeepGEMM/<relative_path>'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "test -f /home/yiqa/DeepGEMM/<relative_path> && docker exec yiqa_deepep bash -lc 'test -f /home/yiqa/DeepGEMM/<relative_path>'"
```

## GPU/DCU State

Check card utilization and memory status inside Docker on both nodes before GPU tests:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; (hy-smi || rocm-smi --showuse --showmemuse || true)'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; (hy-smi || rocm-smi --showuse --showmemuse || true)'"
```

Check device enumeration, ISA, and DTK/ROCm visibility when setting up a node, changing containers, or diagnosing suspicious card state. This is not a replacement for `hy-smi`; it catches "runtime sees no devices / wrong ISA" problems that utilization tables do not show:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; (rocninfo || rocminfo || /opt/dtk/bin/rocminfo) 2>/dev/null | grep -E '\''Name:|Marketing Name:|Vendor Name:|Device Type:|Compute Unit:|SIMDs per CU:|Wavefront Size:|ISA'\'' | head -n 120 || true; ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null || true'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; (rocninfo || rocminfo || /opt/dtk/bin/rocminfo) 2>/dev/null | grep -E '\''Name:|Marketing Name:|Vendor Name:|Device Type:|Compute Unit:|SIMDs per CU:|Wavefront Size:|ISA'\'' | head -n 120 || true; ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null || true'"
```

If there is unexpected VRAM/HCU use, inspect PIDs with `hy-smi --showpids` in Docker and map PIDs on the host with `ps -fp`. Ask before killing unrelated user processes.

Optional package inventory for framework mismatch investigations:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && pip3 list | grep -E '\''torch|sglang|deep|mega|roc|hip'\'' || true'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && pip3 list | grep -E '\''torch|sglang|deep|mega|roc|hip'\'' || true'"
```

## Compile, Test, And Debug Templates

Single-node command pattern, only for diagnostics that run on that same node:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && <cmd>'"
```

Compile smoke on both nodes:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && python3 -m compileall megamoe -q'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && python3 -m compileall megamoe -q'"
```

Source-level pytest or contract tests on both nodes:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py'"
```

Debug script on one or both nodes. Write logs under `hygon_tmp/supernode_debug/` and keep node names in file names:

```powershell
$run = "hygon_tmp/supernode_debug/debug_$(Get-Date -Format yyyyMMdd_HHmmss)"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && mkdir -p $run && PYTHONPATH=. python3 <debug_script.py> <args> 2>&1 | tee $run/node0.log'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && mkdir -p $run && PYTHONPATH=. python3 <debug_script.py> <args> 2>&1 | tee $run/node1.log'"
```

Build on both nodes. Because TX32 has no shared storage, compiled artifacts from node0 are not visible on node1:

```powershell
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && bash ./megamoe/dcu_megamoe_opt/scripts/build_dcu_megamoe.sh'"
ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc 'source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM && bash ./megamoe/dcu_megamoe_opt/scripts/build_dcu_megamoe.sh'"
```

## 32-Card Multi-Node Torchrun Pattern

Use node 0 (`10.17.160.69`) as rendezvous/master unless the user specifies otherwise.

- `MASTER_ADDR=10.17.160.69`
- `MASTER_PORT=<free-port>`
- `NNODES=2`
- SGLang / DeepEP / MegaMoE local-spawn tests: `torchrun --nproc-per-node=1` plus script argument `--num-processes=16`
- Training stacks such as Megatron, or generic one-process-per-device DDP scripts: `torchrun --nproc-per-node=16`
- Node 0 uses `--node-rank=0`
- Node 1 uses `--node-rank=1`
- Each node uses local `HIP_VISIBLE_DEVICES=0,1,2,...,15`

Launch the two commands concurrently from two PowerShell jobs or two terminals.

DeepEP/MegaMoE local-spawn example skeleton:

```powershell
$common = "source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM; export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15; export MASTER_ADDR=10.17.160.69; export MASTER_PORT=29500; torchrun --nnodes=2 --nproc-per-node=1 --master-addr=10.17.160.69 --master-port=29500"
Start-Job -Name tx32-node0 -ScriptBlock { ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc '$using:common --node-rank=0 <script.py> --num-processes 16 <args>'" }
Start-Job -Name tx32-node1 -ScriptBlock { ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc '$using:common --node-rank=1 <script.py> --num-processes 16 <args>'" }
Receive-Job -Name tx32-node0,tx32-node1 -Wait
```

Generic one-process-per-device torchrun skeleton:

```powershell
$common = "source /opt/dtk/env.sh 2>/dev/null || true; cd /home/yiqa/DeepGEMM; export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15; export MASTER_ADDR=10.17.160.69; export MASTER_PORT=29500; torchrun --nnodes=2 --nproc-per-node=16 --master-addr=10.17.160.69 --master-port=29500"
Start-Job -Name tx32-node0 -ScriptBlock { ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.160.69 "docker exec yiqa_deepep bash -lc '$using:common --node-rank=0 <script.py> <args>'" }
Start-Job -Name tx32-node1 -ScriptBlock { ssh -F NUL -p 22 -i C:/Users/Administrator/.ssh/id_rsa root@10.17.162.22 "docker exec yiqa_deepep bash -lc '$using:common --node-rank=1 <script.py> <args>'" }
Receive-Job -Name tx32-node0,tx32-node1 -Wait
```

For long jobs, tee logs into `/home/yiqa/DeepGEMM/hygon_tmp/supernode_debug/<run_name>/node{0,1}.log` inside the container. Keep logs under the repo path.

## Failure Handling

- If SSH fails, verify key auth with `BatchMode=yes` before trying interactive work.
- If `docker exec` fails because the container is stopped, inspect with `docker ps -a --filter name=yiqa_deepep`; starting the existing container is allowed, but do not recreate containers without user confirmation.
- If `docker ps` fails with "Cannot connect to the Docker daemon", inspect only with `systemctl is-active docker`, `systemctl is-active containerd`, `ps -ef | grep -E 'dockerd|containerd|docker'`, and `ls -l /var/run/docker.sock`; ask before starting or restarting Docker because live containers or occupied jobs may be affected.
- If host path exists but Docker path does not, inspect `docker inspect yiqa_deepep --format '{{json .Mounts}}'`.
- If node files differ, resync both nodes and verify checksums of the changed files.
- If a multi-node job hangs, first check that both torchrun commands started, both use the same master address/port, and each uses the correct node rank.
