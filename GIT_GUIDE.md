# humanoid_sim — clone, set up, update, autostart

One place for getting the repo onto a Windows machine, keeping it current from the
remote, and (optionally) auto-starting the boxing game when you log in.

Repo: <https://github.com/flrzep/humanoid_sim>

---

## 0. Prerequisites (install once)

- **Git for Windows** — <https://git-scm.com/download/win>
- **Python 3.10+** — <https://www.python.org/downloads/> (tick *"Add python.exe to PATH"*).
  3.12 / 3.13 are the safest for the `torch` wheel; 3.14 also works.

Check they're on PATH (open a fresh PowerShell):

```powershell
git --version
py --version      # or: python --version
```

---

## 1. Clone the repo

Pick a folder to hold it, then:

```powershell
git clone https://github.com/flrzep/humanoid_sim.git
cd humanoid_sim
```

---

## 2. Set up the Python environment

From inside the repo (this builds `.venv` and installs everything):

```powershell
.\setup.ps1            # or just double-click setup.bat
```

What it does: finds a Python 3.10+, creates `.venv`, installs `requirements.txt`
(mujoco, numpy, scipy, pyyaml, Pillow, torch), and runs an import smoke-test. All
model assets and policy weights are already in the repo, and the browser demos pull
three.js / mujoco-js from a CDN, so there is nothing else to download.

Rebuild the venv from scratch any time with `.\setup.ps1 -Recreate`.

---

## 3. Choose a branch

| Branch | What's on it |
|--------|--------------|
| `main`   | the IMU balancing sim + browser demos |
| `boxing` | the **1v1 boxing game** (this is what most people want) |

```powershell
git checkout boxing
```

The two branches currently share the same dependencies, so you don't need to re-run
`setup.ps1` when switching — but if a future pull changes `requirements.txt`, run it
again (see step 5).

---

## 4. Run it

```powershell
# Boxing game (opens a browser at http://127.0.0.1:8002/):
.\.venv\Scripts\python scripts\boxing_demo.py
.\.venv\Scripts\python scripts\boxing_demo.py --port 8080 --no-browser

# IMU sim in the browser:
.\.venv\Scripts\python scripts\web_demo_wasm.py

# IMU sim in a native viewer with a HUD:
.\.venv\Scripts\python scripts\demo_balance.py

# Run the tests:
.\.venv\Scripts\python -m pytest
```

Stop a running demo with **Ctrl+C** in its window.

---

## 5. Update from the remote

Fetch the latest and fast-forward your current branch:

```powershell
git fetch origin
git status              # shows your branch + whether you have local edits
git pull               # update the branch you're on
```

Update a specific branch:

```powershell
git checkout main   ; git pull
git checkout boxing ; git pull
```

**If `git pull` complains about local changes** you didn't mean to keep, either stash
them temporarily:

```powershell
git stash             # set your changes aside
git pull
git stash pop         # re-apply them on top (resolve conflicts if any)
```

…or throw them away (destructive — you lose those edits):

```powershell
git checkout -- <file>            # discard one file
git reset --hard origin/boxing    # discard ALL local changes on this branch
```

**After pulling, if `requirements.txt` changed**, refresh the environment:

```powershell
.\setup.ps1
```

First time you push/pull on a new machine, Git may prompt for your GitHub login — use
a Personal Access Token as the password (github.com → Settings → Developer settings →
Personal access tokens), or install GitHub CLI (`gh auth login`).

---

## 6. Auto-start the boxing game at logon (optional)

`boxing_autostart.ps1` registers a shortcut in your Windows **Startup** folder so the
boxing server launches (and opens a browser) every time you log in.

```powershell
.\boxing_autostart.ps1 -Install         # turn on autostart at logon
.\boxing_autostart.ps1 -Install -Port 8080 -NoBrowser   # options are remembered
.\boxing_autostart.ps1 -Uninstall       # turn it off
.\boxing_autostart.ps1                   # just run it now (same thing the shortcut does)
```

Notes:
- It runs at **logon** (not at the boot screen), because the demo opens a browser.
- It records the repo's current location. If you **move the repo**, re-run `-Install`.
- The server runs in a minimized PowerShell window — close that window to stop it.
- Run `.\setup.ps1` at least once first, or the launcher will tell you the venv is missing.
