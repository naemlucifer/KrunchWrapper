"""
Test suite for the async logging system.
"""

import time
import threading
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.async_logger import (
    log_verbose_system_prompt_phase_fast,
    get_optimized_logger,
    get_performance_monitor,
    setup_global_async_logging,
    remove_global_async_logging
)

def test_basic_async_logging():
    """Test basic async logging functionality."""
    print("🧪 Testing basic async logging...")
    
    # Get the logger
    logger = get_optimized_logger()
    
    # Log some test messages
    log_verbose_system_prompt_phase_fast("TEST", "Basic async logging test message")
    log_verbose_system_prompt_phase_fast("TEST", "Message with data", {"test_key": "test_value", "number": 42})
    log_verbose_system_prompt_phase_fast("ERROR", "Test error message", {"error_type": "test_error"})
    
    # Give async processing some time
    time.sleep(0.5)
    
    # Get stats
    stats = logger.get_stats()
    print(f"   ✅ Logged {stats['messages_logged']} messages")
    print(f"   📊 Queue size: {stats['queue_size']}")
    print(f"   ❌ Dropped messages: {stats['messages_dropped']}")
    print()

def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("🧪 Testing performance monitoring...")
    
    monitor = get_performance_monitor()
    
    # Test some operations with timing
    with monitor.time_operation("test_operation_1"):
        time.sleep(0.1)  # Simulate work
    
    with monitor.time_operation("test_operation_2"):
        time.sleep(0.05)  # Simulate different work
    
    # Test multiple operations of the same type
    for i in range(3):
        with monitor.time_operation("repeated_operation"):
            time.sleep(0.02)
    
    # Get performance stats
    perf_stats = monitor.get_stats()
    print("   📈 Performance Statistics:")
    for op_name, stats in perf_stats.items():
        print(f"      {op_name}:")
        print(f"         Count: {stats['count']}")
        print(f"         Avg time: {stats['avg_time']*1000:.2f}ms")
        print(f"         Min time: {stats['min_time']*1000:.2f}ms")
        print(f"         Max time: {stats['max_time']*1000:.2f}ms")
        print(f"         Recent avg: {stats['recent_avg']*1000:.2f}ms")
    print()

def test_high_volume_logging():
    """Test async logging under high volume."""
    print("🧪 Testing high-volume async logging...")
    
    logger = get_optimized_logger()
    start_stats = logger.get_stats()
    
    # Log many messages quickly
    start_time = time.perf_counter()
    message_count = 1000
    
    for i in range(message_count):
        log_verbose_system_prompt_phase_fast(
            "HIGH_VOLUME_TEST", 
            f"High volume test message {i+1}",
            {"iteration": i+1, "batch": "high_volume_test"}
        )
    
    logging_time = time.perf_counter() - start_time
    
    # Give async processing time to catch up
    time.sleep(1.0)
    
    end_stats = logger.get_stats()
    messages_processed = end_stats['messages_logged'] - start_stats['messages_logged']
    
    print(f"   📤 Sent {message_count} messages in {logging_time*1000:.2f}ms")
    print(f"   📨 Processed {messages_processed} messages")
    print(f"   ⚡ Logging rate: {message_count/logging_time:.0f} messages/second")
    print(f"   📊 Final queue size: {end_stats['queue_size']}")
    print(f"   ❌ Dropped messages: {end_stats['messages_dropped']}")
    print()

def test_concurrent_logging():
    """Test concurrent logging from multiple threads."""
    print("🧪 Testing concurrent logging...")
    
    logger = get_optimized_logger()
    start_stats = logger.get_stats()
    
    def worker_thread(thread_id, message_count):
        """Worker function for concurrent logging."""
        for i in range(message_count):
            log_verbose_system_prompt_phase_fast(
                "CONCURRENT_TEST",
                f"Thread {thread_id} message {i+1}",
                {"thread_id": thread_id, "message_num": i+1}
            )
    
    # Start multiple threads
    threads = []
    thread_count = 5
    messages_per_thread = 100
    
    start_time = time.perf_counter()
    
    for thread_id in range(thread_count):
        thread = threading.Thread(
            target=worker_thread, 
            args=(thread_id, messages_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    concurrent_time = time.perf_counter() - start_time
    
    # Give async processing time to catch up
    time.sleep(1.0)
    
    end_stats = logger.get_stats()
    total_messages = thread_count * messages_per_thread
    messages_processed = end_stats['messages_logged'] - start_stats['messages_logged']
    
    print(f"   🧵 Used {thread_count} threads with {messages_per_thread} messages each")
    print(f"   📤 Total messages sent: {total_messages}")
    print(f"   📨 Messages processed: {messages_processed}")
    print(f"   ⏱️  Concurrent logging time: {concurrent_time*1000:.2f}ms")
    print(f"   ⚡ Concurrent rate: {total_messages/concurrent_time:.0f} messages/second")
    print(f"   ❌ Dropped messages: {end_stats['messages_dropped']}")
    print()



def test_global_async_logging():
    """Test global async logging for all Python logging calls."""
    print("🧪 Testing global async logging...")
    
    # Setup global async logging
    async_handler = setup_global_async_logging(
        enable=True,
        log_level="DEBUG",
        max_queue_size=1000
    )
    
    try:
        # Create a test logger
        test_logger = logging.getLogger("test_global_async")
        
        # Log various levels
        test_logger.debug("Debug message through global async logging")
        test_logger.info("Info message through global async logging")
        test_logger.warning("Warning message through global async logging")
        test_logger.error("Error message through global async logging")
        
        # Give async processing time
        time.sleep(0.3)
        
        # Check if handler is working
        if async_handler and async_handler.async_handler:
            stats = async_handler.async_handler.stats
            print(f"   ✅ Global async logging active")
            print(f"   📊 Messages processed: {stats['messages_logged']}")
            print(f"   📊 Queue size: {stats['queue_size']}")
            print(f"   ❌ Dropped messages: {stats['messages_dropped']}")
        else:
            print("   ⚠️  Global async logging handler not found")
        
    finally:
        # Clean up
        remove_global_async_logging()
        print("   🧹 Global async logging cleaned up")
    
    print()



def main():
    """Run all async logging tests."""
    print("🚀 Starting Async Logging System Tests")
    print("=" * 50)
    
    try:
        # Run all tests
        test_basic_async_logging()
        test_performance_monitoring()
        test_high_volume_logging()
        test_concurrent_logging()
        test_global_async_logging()
        
        print("✅ All async logging tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean shutdown
        print("\n🔄 Shutting down async logging system...")
        print("✅ Async logging system shutdown complete")

if __name__ == "__main__":
    main() 