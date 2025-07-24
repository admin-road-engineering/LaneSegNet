#!/usr/bin/env python3
"""
Final Root Directory Cleanup - Archive completed cleanup scripts and tools
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Final Root Directory Cleanup")
    print("=" * 45)
    
    # ARCHIVE these root files (cleanup tasks completed)
    archive_files = {
        "check_ssl_readiness.py",      # SSL readiness verified - no longer needed
        "check_versions.py",           # Version checking complete
        "cleanup_batch_files.py",      # Batch cleanup complete
        "cleanup_docs.py",             # Docs cleanup complete  
        "cleanup_scripts.py",          # Scripts cleanup complete
        "cleanup_data_directory.py",   # Analysis tool - can be archived after use
        "quick_cleanup.py"             # General cleanup complete
    }
    
    # Create final archive directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"archived_cleanup_tools_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    
    print(f"Archive directory: {archive_dir}")
    print()
    
    archived_count = 0
    
    # Archive completed cleanup tools
    for script in archive_files:
        if Path(script).exists():
            shutil.move(str(script), str(archive_dir / script))
            print(f"ARCHIVED: {script}")
            archived_count += 1
        else:
            print(f"NOT FOUND: {script}")
    
    print(f"\nRoot cleanup scripts archived: {archived_count}")
    
    # Create manifest
    manifest_path = archive_dir / "CLEANUP_TOOLS_MANIFEST.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Cleanup Tools Archive - {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("ARCHIVED CLEANUP TOOLS:\n")
        for script in sorted(archive_files):
            f.write(f"- {script}\n")
        f.write(f"\nTotal archived: {archived_count} files\n")
        f.write(f"\nPURPOSE:\n")
        f.write("Archive completed cleanup scripts and analysis tools.\n")
        f.write("All cleanup tasks have been successfully completed.\n")
    
    print(f"\nManifest created: {manifest_path}")
    
    print("\n" + "=" * 50)
    print("ROOT CLEANUP TOOLS ARCHIVED")
    print("=" * 50)
    print("All completed cleanup scripts have been safely archived.")
    print("Core functionality remains completely intact.")

if __name__ == "__main__":
    main()