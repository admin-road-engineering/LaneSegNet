#!/usr/bin/env python3
"""
Documentation Cleanup - Archive phase-specific and obsolete documentation
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def main():
    print("LaneSegNet Documentation Cleanup")
    print("=" * 40)
    
    # KEEP these documentation files (current/essential)
    keep_files = {
        "FILE_ORGANIZATION.md",  # Current project structure reference
        "PRODUCTION_DEPLOYMENT.md"  # Production deployment guide - still relevant
    }
    
    # ARCHIVE these documentation files/directories (obsolete/phase-specific)
    archive_items = {
        # Implementation directory - phase-specific planning docs
        "implementation/",
        
        # Reports directory - historical reports and status
        "reports/",  
        
        # Sessions directory - session-specific prompts and planning
        "sessions/",
        
        # Already archived directory - move to new archive
        "archived/"
    }
    
    # Create timestamp for archive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = Path(f"docs_archived_{timestamp}")
    archive_dir.mkdir(exist_ok=True)
    
    print(f"Archive directory: {archive_dir}")
    print()
    
    # Move to docs directory
    docs_path = Path("docs")
    if not docs_path.exists():
        print("ERROR: docs/ directory not found")
        return
    
    os.chdir(docs_path)
    
    archived_count = 0
    
    # Archive directories and files
    for item in archive_items:
        item_path = Path(item)
        if item_path.exists():
            destination = Path("..") / archive_dir / "docs" / item
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item_path), str(destination))
            print(f"ARCHIVED: docs/{item}")
            archived_count += 1
        else:
            print(f"NOT FOUND: docs/{item}")
    
    # Go back to root
    os.chdir("..")
    
    print(f"\nDocumentation items archived: {archived_count}")
    
    # Check what's left in docs/
    remaining_files = []
    if docs_path.exists():
        for item in docs_path.iterdir():
            remaining_files.append(item.name)
    
    print(f"\nREMAINING in docs/:")
    if remaining_files:
        for file in sorted(remaining_files):
            if file in keep_files:
                print(f"  KEPT: {file}")
            else:
                print(f"  UNEXPECTED: {file}")
    else:
        print("  (Directory is empty)")
    
    # Create manifest
    manifest_path = archive_dir / "DOCS_ARCHIVE_MANIFEST.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"Documentation Archive - {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")
        f.write("ARCHIVED ITEMS:\n")
        for item in sorted(archive_items):
            f.write(f"- docs/{item}\n")
        f.write(f"\nKEPT FILES:\n")
        for file in sorted(keep_files):
            f.write(f"- docs/{file}\n")
        f.write(f"\nTotal archived: {archived_count} items\n")
        f.write(f"\nPURPOSE:\n")
        f.write("Archive phase-specific documentation, session prompts, and historical reports\n")
        f.write("while preserving current project structure and deployment documentation.\n")
    
    print(f"\nManifest created: {manifest_path}")
    
    print("\n" + "=" * 50)
    print("DOCUMENTATION CLEANUP COMPLETE")
    print("=" * 50)
    print("docs/ directory now contains only current documentation:")
    for file in sorted(keep_files):
        print(f"- {file}")
    
    print(f"\nHistorical documentation safely archived in: {archive_dir}")

if __name__ == "__main__":
    main()