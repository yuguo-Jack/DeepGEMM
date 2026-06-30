---
name: dcu-supernode-workflow
description: Operate TX32 and other Hygon/DCU scale-up environments for DeepGEMM, MegaMoE, DeepEP, SGLang, or framework tests. Use when Codex needs to inspect DCU state, sync local workspace files to non-shared-storage nodes, run docker-based compile/test/debug/profiling commands, or launch local-spawn and torchrun jobs on TX32 two-node 32-card, single-node 16-card, pod6 40-card, or future DCU supernode clusters.
---

# DCU Supernode Workflow

This is the original TX32 workflow generalized for DCU supernodes. Keep the detailed TX32 command style, but parameterize node count, Docker name, mount path, repo path, DTK path, and SSH route so 16-card single nodes, 32-card TX32, pod6 40-card, and future pods follow the same playbook.

## Execution Contract

- Edit code in the local repository by default.
- Treat remote machines as execution workspaces. Do not rely on remote Git state.
- Most DCU scale-up environments do not share storage. Any required source, config, script, or test file must be synced to every participating node.
- Ask for missing environment parameters before first use of a new environment:
  - node list, node names, and local DCU count per node
  - SSH route: direct or jump host
  - username and authentication method
  - Docker container name
  - host path, container path, and repo path
  - DTK environment path
  - temporary/debug directory preference
- Do not write plaintext passwords, private keys, cookies, or tokens into repository files, skill files, command logs, shell history, or planning files. Prefer SSH keys. If password auth is unavoidable, obtain it at runtime and avoid echoing it.
- Run compile/test/debug/profiling work inside Docker unless the user explicitly requests host-side work.
- Host-side Docker inspection commands such as `docker ps`, `docker inspect`, and `docker start <existing-container>` may run outside Docker.
- Avoid direct host-side compilation/testing outside Docker unless the user explicitly requests it.
- Because many environments use `root`, ask before host-side deletion, recursive move, `chmod`, `chown`, package install, service restart, driver/network edits, Docker recreation, or cleanup outside the repo path.
- Put all remote temporary files, logs, profiler output, and debug artifacts under `<remote_repo>/hygon_tmp/supernode_debug/<env>/<run>/`.
- Do not modify host system files. Do not touch Docker daemon, OS packages, driver files, SSH server config, or network config without explicit user approval.

## Known Profiles

These are topology notes, not secret storage. Passwords are intentionally omitted.

### TX32 Two-Node 32-Card

- Nodes: `10.17.160.69`, `10.17.162.22`
- Login: `root`, normally through `C:/Users/Administrator/.ssh/id_rsa`
- Local DCUs: 16 per node
- Docker: `yiqa_deepep`
- Host/container mount: `/home/yiqa` -> `/home/yiqa`
- Repo path: `/home/yiqa/DeepGEMM`
- Preferred DTK: `/home/yiqa/dtk-26.04.1/env.sh`; fallback `/opt/dtk/env.sh`
- Launch model: treat as two 16-card nodes, not one host with local devices `0..31`

### Single-Node 16-Card

- Node: `10.17.151.1`
- Login: `root`
- Local DCUs: 16
- Docker/mount/repo/DTK: ask the user before first use.
- Launch model: single-node. No multi-node rendezvous is needed unless the target program requires it.

### pod6 40-Card Scale-Up

- Jump host: `10.2.68.128:51730`, user `simsadmin`
- Worker nodes, 4 DCUs each:
  - `c0 172.16.13.166`
  - `c1 172.16.13.112`
  - `c2 172.16.13.147`
  - `c3 172.16.13.89`
  - `c4 172.16.13.107`
  - `c5 172.16.13.106`
  - `c6 172.16.13.161`
  - `c7 172.16.13.108`
  - `c8 172.16.13.98`
  - `c9 172.16.13.95`
- Service/control nodes:
  - `s0 172.16.13.105`
  - `s1 172.16.13.169`
  - `s2 172.16.13.100`
  - `s3 172.16.13.90`
- Docker/mount/repo/DTK: ask the user before first use; pod environments are not guaranteed to match TX32.
- Use jump-host SSH. Do not assume direct routing to `172.16.13.*` from local Windows.
- Do not assume the jump host shares storage with worker nodes. If direct `scp -o ProxyJump=...` is unavailable, copy local -> jump host temporary directory first, then jump host -> every worker node.
- Launch model: ten 4-card nodes. For local-spawn tests, use outer `torchrun --nproc-per-node=1` and script `--num-processes=4`.

### Future Pods

- Treat each future pod as a new profile.
- Do not hardcode old docker names, host paths, container paths, or DTK paths.
- Confirm whether storage is shared. Default assumption: storage is not shared, so sync/build per node.
- Confirm whether the target test uses local-spawn workers or one process per DCU.

## Read Parameters

Read the relevant local environment config first when it exists. For TX32, read `.vscode/tx_supernode32.json`; it should be a JSON array with one entry per node:

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

`context: "."` means the local VS Code workspace root. For non-shared-storage environments, every node entry should point at the same local context and the same intended remote repo path unless the user explicitly says otherwise.

For a new pod, build a small in-memory profile before running commands:

```text
env_name=<short name>
nodes=<node name/ip list>
local_dcus=<count per node>
ssh_args=<direct ssh args or ProxyJump args>
docker_name=<container>
host_repo=<host repo path>
container_repo=<container repo path>
dtk_env=<path to env.sh>
tmp_root=<container_repo>/hygon_tmp/supernode_debug/<env_name>
```

## DTK Environment Loader

Use one generic DTK loader in command templates. Build it from the active profile:

```bash
source_dtk='if [ -n "${DTK_ENV:-}" ] && [ -f "$DTK_ENV" ]; then source "$DTK_ENV"; elif [ -f <profile_dtk_env> ]; then source <profile_dtk_env>; elif [ -f /home/yiqa/dtk-26.04.1/env.sh ]; then source /home/yiqa/dtk-26.04.1/env.sh; elif [ -f /opt/dtk/env.sh ]; then source /opt/dtk/env.sh; else echo "warning: DTK env.sh not found" >&2; fi'
```

In this skill, `<source_dtk>` means the shell snippet above, with `<profile_dtk_env>` replaced by the environment's configured DTK path when known. If a pod has a different DTK install path, update only the profile value, not every command.

## Windows SSH Reliability

- Prefer `ssh -F NUL ...` and `scp -F NUL ...` to avoid local SSH config surprises.
- Use `-o BatchMode=yes` for checks that must prove key-based auth.
- Use `-o StrictHostKeyChecking=accept-new` for first contact with known lab nodes.
- If the private key has Windows ACL issues, copy it to a temporary file with restricted ACLs before retrying.
- If password authentication is required, use an interactive session or a user-provided temporary secure method. Do not place passwords in command strings.

Direct SSH:

```powershell
ssh -F NUL -p <port> -i <key> <user>@<host> "hostname && whoami"
```

Jump-host SSH:

```powershell
ssh -F NUL -o ProxyJump=<jump_user>@<jump_host>:<jump_port> <user>@<node_ip> "hostname && whoami"
```

Direct SCP:

```powershell
scp -F NUL -P <port> -i <key> <local_file> <user>@<host>:<remote_repo>/<relative_path>
```

SCP through jump host:

```powershell
scp -F NUL -o ProxyJump=<jump_user>@<jump_host>:<jump_port> <local_file> <user>@<node_ip>:<remote_repo>/<relative_path>
```

## Quick Verification Workflow

Run these parameterized checks on every node before build/test/launch. Substitute values from the active profile instead of copying an environment-specific block.

Host login and repo path:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "hostname && whoami && test -d <host_repo> && echo host_repo_ok=<host_repo>"
```

Docker and mount:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker ps -a --filter name=<docker_name> --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
ssh -F NUL <ssh_args> <user>@<host> "docker inspect <docker_name> --format '{{json .Mounts}}'"
```

Container repo and toolchain:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; test -d <container_repo> && cd <container_repo> && echo container_repo_ok=<container_repo> && pwd && (which hipcc || true) && python3 -V && pip3 --version'"
```

## Sync Workflow

Treat sync as a two-way workflow:

- Local workspace is the primary editing location.
- Remote nodes are the primary execution locations.
- Non-shared-storage nodes must all receive the same source/config/test files needed by a run.
- For normal editing, prefer VS Code SFTP multi-context config when configured. If upload-on-save is uncertain, explicitly upload touched files or directories.
- Use targeted sync for iterative edits.
- Use full-workspace sync only when setting up a fresh node or after many files changed.
- Do not delete stale remote files unless the user explicitly approves the cleanup and the delete scope is restricted to the repo path.
- Remote-generated logs, profiles, and debug outputs under `hygon_tmp/supernode_debug/` may be copied back to local for analysis.
- Remote temporary/debug artifacts do not all need to be copied back to local. Prefer keeping corresponding run directories organized on each remote node.
- A jump host is not a storage substitute. Syncing files to the jump host alone is insufficient unless the user confirms workers mount the same filesystem.

Verify the repo path exists on every node:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "test -d <host_repo> && echo repo_ok=<host_repo>"
```

Single-file upload:

```powershell
scp -F NUL <ssh_args> <local_file> <user>@<host>:<host_repo>/<relative_path>
```

Directory upload:

```powershell
scp -F NUL <ssh_args> -r <local_dir> <user>@<host>:<host_repo>/<relative_parent>/
```

Two-step upload when ProxyJump `scp` is unavailable or unstable:

```powershell
scp -F NUL -P <jump_port> <local_file> <jump_user>@<jump_host>:<jump_tmp>/<relative_path>
ssh -F NUL -p <jump_port> <jump_user>@<jump_host> "scp <jump_tmp>/<relative_path> <user>@<node_ip>:<host_repo>/<relative_path>"
```

Repeat the final hop for every worker node in the run.

After sync, verify host and container paths see the expected file:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "test -f <host_repo>/<relative_path> && docker exec <docker_name> bash -lc 'test -f <container_repo>/<relative_path>'"
```

Copy selected artifacts back only when local analysis needs them:

```powershell
scp -F NUL <ssh_args> -r <user>@<host>:<container_repo>/hygon_tmp/supernode_debug/<env>/<run> hygon_tmp/supernode_debug/<env>_<node>_<run>
```

## DCU State

Check card utilization and memory status inside Docker on every node before DCU tests:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; (hy-smi || rocm-smi --showuse --showmemuse || true)'"
```

Check active PIDs:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; (hy-smi --showpids || true)'"
```

Check device enumeration, ISA, and runtime visibility when setting up a node, changing containers, or diagnosing suspicious card state:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; (rocninfo || rocminfo || /opt/dtk/bin/rocminfo) 2>/dev/null | grep -E '\''Name:|Marketing Name:|Vendor Name:|Device Type:|Compute Unit:|SIMDs per CU:|Wavefront Size:|ISA'\'' | head -n 120 || true; ls -l /dev/kfd /dev/dri/renderD* 2>/dev/null || true'"
```

Optional package inventory for framework mismatch investigations:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; cd <container_repo> && pip3 list | grep -E '\''torch|sglang|deep|mega|roc|hip'\'' || true'"
```

If there is unexpected VRAM/HCU use, inspect PIDs with `hy-smi --showpids` and map PIDs on the host with `ps -fp`. Ask before killing unrelated user processes.

## Compile, Test, And Debug Templates

Single-node command pattern:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; cd <container_repo> && <cmd>'"
```

Compile smoke on every node:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; cd <container_repo> && python3 -m compileall megamoe -q'"
```

Source-level pytest or contract tests on every node:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; cd <container_repo> && PYTHONPATH=. python3 -m pytest -q megamoe/dcu_megamoe_opt/tests/test_dcu_megamoe_v3.py'"
```

Build on every node that will run compiled artifacts unless storage is confirmed shared:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; cd <container_repo> && mkdir -p hygon_tmp/supernode_debug/<env>/<run> && bash ./megamoe/dcu_megamoe_opt/scripts/build_dcu_megamoe.sh 2>&1 | tee hygon_tmp/supernode_debug/<env>/<run>/<node>_build.log'"
```

Debug script on one or more nodes:

```powershell
ssh -F NUL <ssh_args> <user>@<host> "docker exec <docker_name> bash -lc '<source_dtk>; cd <container_repo> && mkdir -p hygon_tmp/supernode_debug/<env>/<run> && PYTHONPATH=. python3 <debug_script.py> <args> 2>&1 | tee hygon_tmp/supernode_debug/<env>/<run>/<node>.log'"
```

## Launch Model Selection

Select the launch model from the target program, not from card count alone:

- SGLang / DeepEP / MegaMoE local-spawn tests usually use outer `torchrun --nproc-per-node=1`: one launcher process per node, then the script's `--num-processes=<local_dcus>` spawns one local worker per DCU inside that node.
- Training stacks such as Megatron or generic one-process-per-device DDP programs usually use `torchrun --nproc-per-node=<local_dcus>`, because torchrun itself owns the per-DCU worker creation.
- Do not combine `torchrun --nproc-per-node=<local_dcus>` with a script that also does `torch.multiprocessing.spawn(..., nprocs=<local_dcus>)`.

Examples:

- TX32 local-spawn: `--nnodes=2 --nproc-per-node=1`, script `--num-processes=16`.
- TX32 one-process-per-DCU training: `--nnodes=2 --nproc-per-node=16`.
- pod6 local-spawn: `--nnodes=10 --nproc-per-node=1`, script `--num-processes=4`.
- pod6 one-process-per-DCU training: `--nnodes=10 --nproc-per-node=4`.

## Multi-Node Torchrun Pattern

Use one stable worker node as rendezvous/master unless the user specifies otherwise. The same skeleton covers TX32, pod6, and future pods; only `num_nodes`, `local_dcus`, node ranks, and SSH args change.

Common variables:

```powershell
$master = "<master_ip>"
$port = "<free_port>"
$common = "<source_dtk>; cd <container_repo>; export HIP_VISIBLE_DEVICES=<local_device_list>; torchrun --nnodes=<num_nodes> --nproc-per-node=1 --master-addr=$master --master-port=$port"
```

Start one command per node concurrently:

```powershell
Start-Job -Name node0 -ScriptBlock { ssh -F NUL <ssh_args_node0> <user>@<node0> "docker exec <docker_name> bash -lc '$using:common --node-rank=0 <script.py> --num-processes <local_dcus> <args>'" }
Start-Job -Name node1 -ScriptBlock { ssh -F NUL <ssh_args_node1> <user>@<node1> "docker exec <docker_name> bash -lc '$using:common --node-rank=1 <script.py> --num-processes <local_dcus> <args>'" }
Receive-Job -Name node0,node1 -Wait
```

For pod6, extend the same pattern to node ranks `0..9`; use ProxyJump SSH args for each worker.

pod6 40-card local-spawn example, for DeepEP/MegaMoE/SGLang-style tests where the script spawns 4 local workers per node:

```powershell
$nodes = @(
  "172.16.13.166",  # c0, rank 0, rendezvous/master
  "172.16.13.112",  # c1
  "172.16.13.147",  # c2
  "172.16.13.89",   # c3
  "172.16.13.107",  # c4
  "172.16.13.106",  # c5
  "172.16.13.161",  # c6
  "172.16.13.108",  # c7
  "172.16.13.98",   # c8
  "172.16.13.95"    # c9
)
$jump = "-o ProxyJump=simsadmin@10.2.68.128:51730"
$workerUser = "<worker_user>"
$docker = "<docker_name>"
$repo = "<container_repo>"
$sourceDtk = "<source_dtk>"
$master = $nodes[0]
$port = "29500"
$common = "$sourceDtk; cd $repo; export HIP_VISIBLE_DEVICES=0,1,2,3; torchrun --nnodes=10 --nproc-per-node=1 --master-addr=$master --master-port=$port"

for ($rank = 0; $rank -lt $nodes.Count; $rank++) {
  $node = $nodes[$rank]
  Start-Job -Name "pod6-rank$rank" -ArgumentList $node,$rank,$jump,$workerUser,$docker,$common -ScriptBlock {
    param($node, $rank, $jump, $workerUser, $docker, $common)
    ssh -F NUL $jump "$workerUser@$node" "docker exec $docker bash -lc '$common --node-rank=$rank <script.py> --num-processes 4 <args>'"
  }
}
Receive-Job -Name (0..9 | ForEach-Object { "pod6-rank$_" }) -Wait
```

For one-process-per-DCU training on pod6, change `$common` to use `--nproc-per-node=4` and remove the script-level `--num-processes 4`.

For long jobs, tee logs into `<container_repo>/hygon_tmp/supernode_debug/<env>/<run>/node{rank}.log` inside the container.

## Failure Handling

- If SSH fails, distinguish local routing, jump-host login, target login, and Docker/container failures.
- If SSH key auth should work, verify with `BatchMode=yes` before trying interactive work.
- If `docker exec` fails because the container is stopped, inspect with `docker ps -a --filter name=<docker_name>`. Starting an existing container is allowed only when it will not disturb other users.
- If `docker ps` fails with "Cannot connect to the Docker daemon", inspect only with `systemctl is-active docker`, `systemctl is-active containerd`, `ps -ef | grep -E 'dockerd|containerd|docker'`, and `ls -l /var/run/docker.sock`; ask before starting or restarting Docker because live containers or occupied jobs may be affected.
- If host path exists but Docker path does not, inspect `docker inspect <docker_name> --format '{{json .Mounts}}'`.
- If node files differ, resync changed files to every node and verify checksums.
- If a multi-node job hangs, first check that all torchrun commands started, all use the same master address/port, every node rank is unique, and no script is double-spawning local DCU workers.
- If cards stay occupied by unrelated processes, report the PIDs and wait or ask. Do not kill without explicit approval.
