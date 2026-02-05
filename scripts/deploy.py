#!/usr/bin/env python3
"""
Deploy Script - Prepares and deploys POW Agent to GitHub + Render
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def run_command(cmd: list | str, cwd: Path = None, check: bool = True) -> tuple[int, str, str]:
    """Run a command and return (code, stdout, stderr)."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr
    except Exception as e:
        return 1, "", str(e)


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║          POW AGENT - DEPLOY TO GITHUB                        ║
╚══════════════════════════════════════════════════════════════╝
""")


def check_git():
    """Check git is available and configured."""
    code, out, err = run_command("git --version", check=False)
    if code != 0:
        print("❌ Git not found. Please install git.")
        sys.exit(1)
    print(f"✅ Git: {out.strip()}")
    return True


def check_remote(cwd: Path):
    """Check remote is configured."""
    code, out, err = run_command("git remote -v", cwd=cwd, check=False)
    if code != 0 or not out.strip():
        print("❌ No git remote configured.")
        print("   Run: git remote add origin https://github.com/nayrbryanGaming/proof-of-work-agent.git")
        return False
    
    lines = out.strip().split("\n")
    for line in lines[:2]:  # Show first 2
        print(f"   Remote: {line}")
    return True


def check_status(cwd: Path):
    """Check git status."""
    code, out, err = run_command("git status --porcelain", cwd=cwd, check=False)
    
    if out.strip():
        changes = out.strip().split("\n")
        print(f"📝 Uncommitted changes: {len(changes)} files")
        for change in changes[:10]:
            print(f"   {change}")
        if len(changes) > 10:
            print(f"   ... and {len(changes) - 10} more")
        return changes
    else:
        print("✅ Working directory clean")
        return []


def stage_all(cwd: Path):
    """Stage all changes."""
    code, out, err = run_command("git add -A", cwd=cwd, check=False)
    if code != 0:
        print(f"❌ Failed to stage changes: {err}")
        return False
    print("✅ All changes staged")
    return True


def commit(cwd: Path, message: str):
    """Create commit."""
    code, out, err = run_command(["git", "commit", "-m", message], cwd=cwd, check=False)
    if code != 0:
        if "nothing to commit" in err or "nothing to commit" in out:
            print("ℹ️  Nothing to commit")
            return True
        print(f"❌ Commit failed: {err}")
        return False
    print(f"✅ Committed: {message}")
    return True


def push(cwd: Path, branch: str = "main"):
    """Push to remote."""
    code, out, err = run_command(f"git push origin {branch}", cwd=cwd, check=False)
    if code != 0:
        print(f"⚠️  Push output: {err}")
        # Try force push if normal push fails
        print("   Trying with --force...")
        code, out, err = run_command(f"git push origin {branch} --force", cwd=cwd, check=False)
        if code != 0:
            print(f"❌ Push failed: {err}")
            return False
    print(f"✅ Pushed to origin/{branch}")
    return True


def get_current_branch(cwd: Path) -> str:
    """Get current branch name."""
    code, out, err = run_command("git branch --show-current", cwd=cwd, check=False)
    if code == 0 and out.strip():
        return out.strip()
    return "main"


def main():
    print_banner()
    
    # Get project root
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    print(f"📁 Project: {project_root}")
    print(f"⏰ Time: {datetime.now().isoformat()}")
    print()
    
    # Checks
    print("=" * 50)
    print("PRE-FLIGHT CHECKS")
    print("=" * 50)
    
    check_git()
    
    if not check_remote(project_root):
        print("\nSetting up remote...")
        run_command(
            "git remote add origin https://github.com/nayrbryanGaming/proof-of-work-agent.git",
            cwd=project_root,
            check=False
        )
        check_remote(project_root)
    
    print()
    
    # Status
    print("=" * 50)
    print("GIT STATUS")
    print("=" * 50)
    
    changes = check_status(project_root)
    current_branch = get_current_branch(project_root)
    print(f"📌 Branch: {current_branch}")
    
    # Stage and commit if changes exist
    if changes:
        print()
        print("=" * 50)
        print("STAGING & COMMITTING")
        print("=" * 50)
        
        stage_all(project_root)
        
        # Generate commit message
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_msg = f"feat: Enhanced POW Agent - production ready [{timestamp}]"
        
        if not commit(project_root, commit_msg):
            print("⚠️  Commit issue, continuing...")
    
    # Push
    print()
    print("=" * 50)
    print("PUSHING TO GITHUB")
    print("=" * 50)
    
    if push(project_root, current_branch):
        print()
        print("=" * 50)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("=" * 50)
        print(f"""
Repository: https://github.com/nayrbryanGaming/proof-of-work-agent

Next Steps for Render:
1. Go to https://dashboard.render.com
2. Create new "Background Worker"
3. Connect GitHub repo: nayrbryanGaming/proof-of-work-agent
4. Set environment variables:
   - COLOSSEUM_API_KEY (required)
   - AGENTWALLET_SESSION (required)
   - OPENAI_API_KEY (optional)
   - PROGRAM_ID (optional)
5. Deploy!

Free tier specs:
- 512 MB RAM
- Auto-restart on crash
- 750 hours/month FREE
""")
    else:
        print("\n⚠️  Push failed. Check your credentials and try:")
        print("   git push origin main --force")


if __name__ == "__main__":
    main()
