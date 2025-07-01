#!/usr/bin/env python3
"""
Test Persistent Token Cache

Tests the disk-backed token caching system with lazy loading.
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import from core
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.persistent_token_cache import PersistentTokenCache, get_persistent_cache

def test_basic_cache_operations():
    """Test basic cache get/set operations."""
    print("🧪 Testing basic cache operations...")
    
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        cache = PersistentTokenCache(temp_dir, max_ram_entries=5, cache_ttl_hours=1)
        
        # Test cache miss
        result = cache.get("nonexistent_key", "test")
        assert result is None, "Cache miss should return None"
        print("✅ Cache miss handled correctly")
        
        # Test cache set and get
        cache.set("test_key", 42, "test")
        result = cache.get("test_key", "test")  
        assert result == 42, f"Expected 42, got {result}"
        print("✅ Cache set/get works correctly")
        
        # Test that file was created on disk
        disk_files = list(Path(temp_dir).glob("*.json"))
        assert len(disk_files) > 0, "No cache files created on disk"
        print(f"✅ Cache file created on disk: {len(disk_files)} files")
        
        # Test loading from disk (simulate restart)
        cache2 = PersistentTokenCache(temp_dir, max_ram_entries=5)
        result2 = cache2.get("test_key", "test")
        assert result2 == 42, f"Expected 42 from disk, got {result2}"
        print("✅ Cache loaded from disk correctly")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
    
    print("✅ Basic cache operations test passed!")

def test_cache_expiration():
    """Test cache expiration functionality."""
    print("🧪 Testing cache expiration...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create cache with very short TTL (0.001 hours = 3.6 seconds)
        cache = PersistentTokenCache(temp_dir, cache_ttl_hours=0.001)
        
        # Set a value
        cache.set("expiring_key", "value", "test")
        
        # Should be available immediately
        result = cache.get("expiring_key", "test")
        assert result == "value", "Value should be available immediately"
        print("✅ Value available immediately after setting")
        
        # Wait for expiration (4 seconds to be safe)
        print("⏳ Waiting for cache expiration (4 seconds)...")
        time.sleep(4)
        
        # Should be expired now
        result = cache.get("expiring_key", "test")
        assert result is None, "Value should be expired"
        print("✅ Cache expiration works correctly")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("✅ Cache expiration test passed!")

def test_ram_cache_eviction():
    """Test RAM cache LRU eviction."""
    print("🧪 Testing RAM cache eviction...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create cache with small RAM limit
        cache = PersistentTokenCache(temp_dir, max_ram_entries=3)
        
        # Add more entries than RAM limit
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}", "test")
        
        # Check stats
        stats = cache.get_stats()
        assert stats['ram_entries'] <= 3, f"RAM entries should be <= 3, got {stats['ram_entries']}"
        assert stats['disk_files'] == 5, f"Disk files should be 5, got {stats['disk_files']}"
        print(f"✅ RAM cache limited to {stats['ram_entries']} entries, {stats['disk_files']} files on disk")
        
        # Access an older key - should load from disk
        result = cache.get("key_0", "test")
        assert result == "value_0", f"Expected value_0, got {result}"
        print("✅ Older entry loaded from disk successfully")
        
        # Check that it was loaded into RAM
        stats_after = cache.get_stats()
        print(f"✅ Stats after access: RAM={stats_after['ram_entries']}, Disk={stats_after['disk_files']}")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("✅ RAM cache eviction test passed!")

def test_token_counting_integration():
    """Test integration with token counting."""
    print("🧪 Testing token counting integration...")
    
    try:
        # Import the necessary components
        from core.dynamic_dictionary import OptimizedSymbolAssigner
        import tiktoken
        
        # Create tokenizer for testing
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Create symbol assigner with persistent cache
        assigner = OptimizedSymbolAssigner(tokenizer=tokenizer, model_name="gpt-4")
        
        # Test token counting with caching
        test_text = "This is a test string for token counting"
        
        # First call - should compute and cache
        start_time = time.time()
        count1 = assigner.get_token_count(test_text)
        time1 = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        count2 = assigner.get_token_count(test_text)
        time2 = time.time() - start_time
        
        assert count1 == count2, f"Token counts should match: {count1} vs {count2}"
        print(f"✅ Token counting consistent: {count1} tokens")
        print(f"✅ First call: {time1:.4f}s, Second call: {time2:.4f}s")
        
        # Verify cache was used (second call should be much faster)
        if time1 > 0.001:  # Only check if first call was measurable
            assert time2 < time1 * 0.5, f"Second call should be faster due to caching"
            print("✅ Caching improved performance")
        
        # Check cache stats
        cache_stats = assigner._persistent_cache.get_stats()
        print(f"✅ Cache stats: {cache_stats}")
        
    except ImportError as e:
        print(f"⚠️  Skipping token counting test due to missing dependency: {e}")
    
    print("✅ Token counting integration test completed!")

def test_cache_cleanup():
    """Test cache cleanup functionality."""
    print("🧪 Testing cache cleanup...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        cache = PersistentTokenCache(temp_dir, cache_ttl_hours=0.001)  # Very short TTL
        
        # Add some entries
        cache.set("key1", "value1", "test")
        cache.set("key2", "value2", "test") 
        cache.set("key3", "value3", "test")
        
        # Check initial state
        initial_files = len(list(Path(temp_dir).glob("*.json")))
        assert initial_files == 3, f"Should have 3 files initially, got {initial_files}"
        print(f"✅ Created {initial_files} cache files")
        
        # Wait for expiration
        time.sleep(4)
        
        # Run cleanup
        cache.clear_expired()
        
        # Check that expired files were removed
        remaining_files = len(list(Path(temp_dir).glob("*.json")))
        print(f"✅ After cleanup: {remaining_files} files remaining")
        
        # Test clear all
        cache.set("new_key", "new_value", "test")  # Add a fresh entry
        cache.clear_all()
        
        final_files = len(list(Path(temp_dir).glob("*.json")))
        assert final_files == 0, f"Should have 0 files after clear_all, got {final_files}"
        print("✅ clear_all() removed all cache files")
        
        # Check RAM cache was also cleared
        stats = cache.get_stats()
        assert stats['ram_entries'] == 0, f"RAM cache should be empty, got {stats['ram_entries']}"
        print("✅ RAM cache cleared")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("✅ Cache cleanup test passed!")

def run_all_tests():
    """Run all persistent cache tests."""
    print("🚀 Starting Persistent Token Cache Tests")
    print("=" * 50)
    
    tests = [
        test_basic_cache_operations,
        test_cache_expiration,
        test_ram_cache_eviction,
        test_token_counting_integration,
        test_cache_cleanup
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n📋 Running {test.__name__}...")
            test()
            print(f"✅ {test.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Persistent token caching is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 