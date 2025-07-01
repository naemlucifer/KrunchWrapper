#!/usr/bin/env python
"""Test the enhanced compression with the new 500-token Python dictionary."""
import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.compress import compress, decompress

def test_enhanced_compression():
    """Test compression with the enhanced dictionary."""
    print("🚀 Testing Enhanced Compression with Token Dictionary")
    print("=" * 65)
    
    # Ensure temp directory exists
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Use files from this project instead of hardcoded paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Find Python files in the project
    test_files = []
    for root_dir in ["core", "scripts", "tests", "utils"]:
        dir_path = os.path.join(project_root, root_dir)
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        test_files.append(os.path.join(root, file))
    
    # Sort by size and take the 10 largest
    test_files.sort(key=lambda f: os.path.getsize(f) if os.path.exists(f) else 0, reverse=True)
    test_files = test_files[:10]
    
    total_original = 0
    total_compressed = 0
    
    for i, file_path in enumerate(test_files):
        if not os.path.exists(file_path):
            print(f"   ⚠️  File {i+1} not found: {os.path.basename(file_path)}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compress
            compressed = compress(content, lang='python')
            
            # Verify roundtrip
            decompressed = decompress(compressed.text, compressed.used)
            roundtrip_success = decompressed == content
            
            # Calculate stats
            original_size = len(content)
            compressed_size = len(compressed.text)
            compression_ratio = (1 - compressed_size / original_size) * 100
            chars_saved = original_size - compressed_size
            
            total_original += original_size
            total_compressed += compressed_size
            
            filename = os.path.basename(file_path)
            base_name = os.path.splitext(filename)[0]
            
            # Save original file to temp
            original_temp_path = os.path.join(temp_dir, f"{i+1:02d}_original_{base_name}.py")
            with open(original_temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Save compressed file to temp
            compressed_temp_path = os.path.join(temp_dir, f"{i+1:02d}_compressed_{base_name}.txt")
            with open(compressed_temp_path, 'w', encoding='utf-8') as f:
                f.write(compressed.text)
            
            # Save token mapping to temp
            tokens_temp_path = os.path.join(temp_dir, f"{i+1:02d}_tokens_{base_name}.json")
            import json
            with open(tokens_temp_path, 'w', encoding='utf-8') as f:
                json.dump(compressed.used, f, indent=2, ensure_ascii=False)
            
            print(f"\\n📄 File {i+1}: {filename}")
            print(f"   Original size:     {original_size:,} characters")
            print(f"   Compressed size:   {compressed_size:,} characters")
            print(f"   Compression ratio: {compression_ratio:.1f}%")
            print(f"   Characters saved:  {chars_saved:,}")
            print(f"   Tokens used:       {len(compressed.used)}")
            print(f"   Roundtrip:         {'✅ Success' if roundtrip_success else '❌ Failed'}")
            print(f"   💾 Saved to temp:   {original_temp_path}")
            print(f"   💾 Compressed:      {compressed_temp_path}")
            print(f"   💾 Tokens:          {tokens_temp_path}")
            
            # Show top tokens used
            if compressed.used:
                print(f"   Top tokens used:")
                # Count token usage
                token_counts = {}
                for short_token, long_token in compressed.used.items():
                    count = compressed.text.count(short_token)
                    token_counts[long_token] = count
                sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
                for j, (token, count) in enumerate(sorted_tokens[:5]):
                    print(f"     {j+1}. '{token}' used {count} times")
        
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
    
    # Overall statistics
    if total_original > 0:
        overall_compression = (1 - total_compressed / total_original) * 100
        overall_saved = total_original - total_compressed
        
        # Save overall results to temp
        results_path = os.path.join(temp_dir, "compression_results_summary.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("Enhanced Compression Test Results\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Total files processed: {len([f for f in test_files if os.path.exists(f)])}\\n")
            f.write(f"Total original size:   {total_original:,} characters\\n")
            f.write(f"Total compressed size: {total_compressed:,} characters\\n")
            f.write(f"Overall compression:   {overall_compression:.1f}%\\n")
            f.write(f"Total chars saved:     {overall_saved:,} characters\\n")
            f.write(f"\\nThis represents a {overall_compression:.1f}% reduction in file size!\\n")
        
        print(f"\\n📊 Overall Results:")
        print(f"   Total original:    {total_original:,} characters")
        print(f"   Total compressed:  {total_compressed:,} characters")
        print(f"   Overall compression: {overall_compression:.1f}%")
        print(f"   Total chars saved: {overall_saved:,}")
        print(f"   💾 Summary saved:   {results_path}")
        print(f"\\n🎯 This represents a {overall_compression:.1f}% reduction in file size!")

if __name__ == "__main__":
    test_enhanced_compression() 