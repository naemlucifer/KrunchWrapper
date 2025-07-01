#!/bin/bash

# KrunchWrapper Cleanup Script
# Removes all contents from logs/ and temp/ folders

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🧹 KrunchWrapper Cleanup Script"
echo "Working directory: $(pwd)"
echo "=" * 50

# Function to safely remove folder contents
cleanup_folder() {
    local folder=$1
    if [ -d "$folder" ]; then
        echo "🗑️  Cleaning $folder..."
        file_count=$(find "$folder" -type f 2>/dev/null | wc -l)
        if [ "$file_count" -gt 0 ]; then
            echo "   Found $file_count files to remove"
            sudo rm -rf "$folder"/*
            echo "   ✅ Cleaned $folder (removed $file_count files)"
        else
            echo "   ℹ️  $folder is already empty"
        fi
    else
        echo "   ⚠️  $folder does not exist, creating it..."
        mkdir -p "$folder"
        echo "   ✅ Created $folder"
    fi
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "core" ]; then
    echo "❌ Error: This doesn't appear to be the KrunchWrapper root directory"
    echo "   Expected to find README.md and core/ folder"
    exit 1
fi

echo "🔍 Verifying KrunchWrapper project structure..."
echo "   ✅ Found README.md"
echo "   ✅ Found core/ directory"

# Clean up logs folder
cleanup_folder "logs"

# Clean up temp folder  
cleanup_folder "temp"

# Also clean any .log files in root directory
echo "🗑️  Cleaning .log files in root directory..."
log_files=$(find . -maxdepth 1 -name "*.log" -type f 2>/dev/null | wc -l)
if [ "$log_files" -gt 0 ]; then
    echo "   Found $log_files .log files to remove"
    sudo rm -f *.log
    echo "   ✅ Removed $log_files .log files from root"
else
    echo "   ℹ️  No .log files found in root directory"
fi

# Optional: Clean Python cache files
echo "🗑️  Cleaning Python cache files..."
cache_dirs=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
if [ "$cache_dirs" -gt 0 ]; then
    echo "   Found $cache_dirs __pycache__ directories to remove"
    find . -name "__pycache__" -type d -exec sudo rm -rf {} + 2>/dev/null || true
    echo "   ✅ Removed Python cache directories"
else
    echo "   ℹ️  No __pycache__ directories found"
fi

# Clean pytest cache
echo "🗑️  Cleaning pytest cache..."
pytest_cache_dirs=0
if [ -d ".pytest_cache" ]; then
    pytest_cache_dirs=$((pytest_cache_dirs + 1))
fi
if [ -d ".cache" ]; then
    pytest_cache_dirs=$((pytest_cache_dirs + 1))
fi

if [ "$pytest_cache_dirs" -gt 0 ]; then
    echo "   Found $pytest_cache_dirs pytest cache directories to remove"
    [ -d ".pytest_cache" ] && sudo rm -rf ".pytest_cache" && echo "   ✅ Removed .pytest_cache/"
    [ -d ".cache" ] && sudo rm -rf ".cache" && echo "   ✅ Removed .cache/"
else
    echo "   ℹ️  No pytest cache directories found"
fi

echo ""
echo "✅ Cleanup completed successfully!"
echo "🔍 Summary:"
echo "   • Cleaned logs/ folder"
echo "   • Cleaned temp/ folder" 
echo "   • Removed .log files from root"
echo "   • Removed Python cache files"
echo "   • Removed pytest cache directories"
echo ""
echo "💡 Tip: Run this script anytime to clean up KrunchWrapper generated files" 