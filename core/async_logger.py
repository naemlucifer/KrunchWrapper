"""
High-performance async logging system for minimal performance impact.
"""

import asyncio
import logging
import queue
import threading
import time
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json

# Logger for async logging setup messages
async_setup_logger = logging.getLogger('async_logger')

@dataclass
class LogEntry:
    """Structured log entry for efficient processing."""
    timestamp: float
    level: str
    phase: str
    message: str
    data: Optional[Any] = None
    correlation_id: Optional[str] = None

def load_async_logging_config() -> Dict[str, Any]:
    """Load async logging configuration from server.jsonc (single source of truth)."""
    try:
        # Import here to avoid circular dependencies
        sys.path.append(str(Path(__file__).parent.parent / "utils"))
        from jsonc_parser import load_jsonc
        
        # Load server config - now the single source of truth for ALL logging settings
        server_config_path = Path(__file__).parent.parent / "config" / "server.jsonc"
        
        if server_config_path.exists():
            try:
                server_config = load_jsonc(server_config_path)
                logging_section = server_config.get("logging", {})
                
                async_setup_logger.info(f"📋 Loading unified logging config from: {server_config_path}")
                
                # Extract all async logging settings from server config
                config = {
                    "enabled": logging_section.get("async_logging_enabled", True),
                    "log_level": logging_section.get("log_level", "INFO"),
                    "max_queue_size": logging_section.get("async_max_queue_size", 0),
                    "batch_size": logging_section.get("async_batch_size", 50),
                    "worker_timeout": logging_section.get("async_worker_timeout", 0.1),
                    "performance_monitoring": {
                        "enabled": logging_section.get("async_performance_monitoring_enabled", True),
                        "track_system_prompts": logging_section.get("async_track_system_prompts", True),
                        "track_compression": logging_section.get("async_track_compression", True),
                        "max_tracked_operations": logging_section.get("async_max_tracked_operations", 100)
                    }
                }
                
                # Async config loaded (shown in formatted section)
                return config
                
            except Exception as e:
                async_setup_logger.warning(f"⚠️ Failed to read unified logging config from server config, using defaults: {e}")
        else:
            async_setup_logger.warning(f"⚠️ Server config not found: {server_config_path}")
        
    except Exception as e:
        async_setup_logger.warning(f"⚠️ Failed to load unified logging config: {e}")
    
    # Default configuration
    async_setup_logger.info("📋 Using default async logging configuration")
    return {
        "enabled": True,
        "log_level": "INFO",
        "max_queue_size": 0,  # Unlimited
        "batch_size": 50,
        "worker_timeout": 0.1,
        "use_optimized_logger_only": True,  # Use new high-performance approach
        "performance_monitoring": {
            "enabled": True,
            "track_system_prompts": True,
            "track_compression": True,
            "max_tracked_operations": 100
        }
    }

def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to unified config."""
    # Override async logging settings via environment variables
    # NOTE: Log level is controlled by KRUNCHWRAPPER_LOG_LEVEL in the main server config
    
    if os.environ.get("KRUNCHWRAPPER_GLOBAL_ASYNC_LOGGING"):
        config["enabled"] = os.environ.get("KRUNCHWRAPPER_GLOBAL_ASYNC_LOGGING", "true").lower() in ("true", "1", "yes")
        async_setup_logger.info(f"📋 Environment override: async_logging_enabled = {config['enabled']}")
    
    if os.environ.get("KRUNCHWRAPPER_ASYNC_QUEUE_SIZE"):
        try:
            config["max_queue_size"] = int(os.environ.get("KRUNCHWRAPPER_ASYNC_QUEUE_SIZE"))
            async_setup_logger.info(f"📋 Environment override: async_max_queue_size = {config['max_queue_size']}")
        except ValueError:
            async_setup_logger.warning(f"⚠️ Invalid KRUNCHWRAPPER_ASYNC_QUEUE_SIZE value")
    
    if os.environ.get("KRUNCHWRAPPER_ASYNC_BATCH_SIZE"):
        try:
            config["batch_size"] = int(os.environ.get("KRUNCHWRAPPER_ASYNC_BATCH_SIZE"))
            async_setup_logger.info(f"📋 Environment override: async_batch_size = {config['batch_size']}")
        except ValueError:
            async_setup_logger.warning(f"⚠️ Invalid KRUNCHWRAPPER_ASYNC_BATCH_SIZE value")
    
    if os.environ.get("KRUNCHWRAPPER_ASYNC_WORKER_TIMEOUT"):
        try:
            config["worker_timeout"] = float(os.environ.get("KRUNCHWRAPPER_ASYNC_WORKER_TIMEOUT"))
            async_setup_logger.info(f"📋 Environment override: async_worker_timeout = {config['worker_timeout']}")
        except ValueError:
            async_setup_logger.warning(f"⚠️ Invalid KRUNCHWRAPPER_ASYNC_WORKER_TIMEOUT value")
    
    return config

class AsyncLogHandler:
    """Non-blocking log handler with configurable batching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = load_async_logging_config()
            config = apply_environment_overrides(config)
        
        self.config = config
        
        # Handle unlimited queue size (0 means unlimited)
        max_queue_size = self.config.get("max_queue_size", 0)
        if max_queue_size == 0:
            # Use a very large number for "unlimited"
            max_queue_size = float('inf')
            self.log_queue = queue.Queue()  # No maxsize for unlimited
        else:
            self.log_queue = queue.Queue(maxsize=max_queue_size)
        
        self.worker_thread = None
        self.running = False
        self.batch_size = self.config.get("batch_size", 50)
        self.worker_timeout = self.config.get("worker_timeout", 0.1)
        
        self.stats = {
            'messages_logged': 0,
            'messages_dropped': 0,
            'queue_size': 0,
            'batches_processed': 0,
            'avg_batch_size': 0
        }
        
        # Performance monitoring
        perf_config = self.config.get("performance_monitoring", {})
        self.performance_monitoring = perf_config.get("enabled", True)
        self.track_system_prompts = perf_config.get("track_system_prompts", True)
        self.track_compression = perf_config.get("track_compression", True)
        self.max_tracked_operations = perf_config.get("max_tracked_operations", 100)
        
        if self.performance_monitoring:
            self.operation_times = {}
            self.operation_counts = {}
        
    def start(self):
        """Start background logging worker."""
        if self.running:
            return
            
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def stop(self):
        """Stop background logging worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)
    
    def log_async(self, 
                  phase: str, 
                  message: str, 
                  data: Optional[Any] = None,
                  level: str = "INFO",
                  correlation_id: Optional[str] = None):
        """Log message asynchronously without blocking."""
        
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            phase=phase,
            message=message,
            data=data,
            correlation_id=correlation_id
        )
        
        try:
            self.log_queue.put_nowait(entry)
            self.stats['messages_logged'] += 1
            
            # Track performance if enabled
            if self.performance_monitoring:
                self._track_operation(phase, entry.timestamp)
                
        except queue.Full:
            self.stats['messages_dropped'] += 1
            # Optionally log to stderr for critical messages
            if level in ['ERROR', 'CRITICAL']:
                print(f"DROPPED LOG: [{phase}] {message}", file=sys.stderr)
    
    def _track_operation(self, operation_name: str, timestamp: float):
        """Track operation timing for performance monitoring."""
        if not self.performance_monitoring:
            return
            
        # Only track specified operation types
        track_this = False
        if self.track_system_prompts and "SYSTEM_PROMPT" in operation_name:
            track_this = True
        elif self.track_compression and any(keyword in operation_name.upper() for keyword in ["COMPRESS", "DECOMPRESS", "DYNAMIC"]):
            track_this = True
        
        if not track_this:
            return
            
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        # Store timestamp for later processing
        self.operation_times[operation_name].append(timestamp)
        self.operation_counts[operation_name] += 1
        
        # Keep only recent measurements to limit memory
        if len(self.operation_times[operation_name]) > self.max_tracked_operations:
            self.operation_times[operation_name] = self.operation_times[operation_name][-self.max_tracked_operations//2:]
    
    def _worker(self):
        """Background worker that processes log entries in batches."""
        batch = []
        
        while self.running:
            try:
                # Collect entries up to batch_size
                batch_start_time = time.time()
                
                while len(batch) < self.batch_size and (time.time() - batch_start_time) < self.worker_timeout:
                    try:
                        # Use a shorter timeout for individual gets
                        entry = self.log_queue.get(timeout=min(0.01, self.worker_timeout / 10))
                        batch.append(entry)
                    except queue.Empty:
                        break
                
                # Process batch if we have entries
                if batch:
                    self._process_batch(batch)
                    self.stats['batches_processed'] += 1
                    
                    # Update average batch size
                    total_entries = self.stats['batches_processed'] * self.stats.get('avg_batch_size', 0) + len(batch)
                    self.stats['avg_batch_size'] = total_entries / (self.stats['batches_processed'] + 1) if self.stats['batches_processed'] > 0 else len(batch)
                    
                    batch.clear()
                
                # Update queue size stats
                self.stats['queue_size'] = self.log_queue.qsize()
                
            except Exception as e:
                # Only log to stderr to avoid circular calls
                print(f"Async logger worker error: {e}", file=sys.stderr)
                # Clear batch on error to prevent infinite loops
                batch.clear()
    
    def _process_batch(self, batch: List[LogEntry]):
        """Process a batch of log entries."""
        # For now, just update statistics
        # In the future, this could write to files, send to external systems, etc.
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance monitoring statistics."""
        if not self.performance_monitoring:
            return {"performance_monitoring": "disabled"}
        
        stats = {}
        for op_name in self.operation_times:
            timestamps = self.operation_times[op_name]
            if len(timestamps) > 1:
                # Calculate operation frequency
                time_span = timestamps[-1] - timestamps[0]
                frequency = len(timestamps) / max(time_span, 1)  # ops per second
                
                stats[op_name] = {
                    'count': self.operation_counts[op_name],
                    'frequency_per_sec': frequency,
                    'recent_count': len(timestamps),
                    'time_span': time_span
                }
        return stats

class OptimizedSystemPromptLogger:
    """Optimized logging for high-performance async operations."""
    
    def __init__(self, enable_verbose: bool = True, config: Optional[Dict[str, Any]] = None):
        self.enable_verbose = enable_verbose
        self.async_handler = AsyncLogHandler(config)
        self.correlation_counter = 0
        
        # Get standard logging setup for actual output
        self.standard_logger = logging.getLogger('krunchwrapper.async')
        
        if enable_verbose:
            self.async_handler.start()
    
    def log_phase(self, 
                  phase: str, 
                  message: str, 
                  data: Optional[Any] = None,
                  correlation_id: Optional[str] = None,
                  level: str = "INFO"):
        """Log a processing phase with high performance async logging."""
        
        if not self.enable_verbose:
            return
        
        # Generate correlation ID if not provided
        if correlation_id is None:
            self.correlation_counter += 1
            correlation_id = f"async_{self.correlation_counter}"
        
        # Queue async performance tracking (non-blocking)
        self.async_handler.log_async(
            phase=f"{phase}",
            message=message,
            data=data,
            correlation_id=correlation_id,
            level=level
        )
        
        # Emit actual log message through standard logging (benefits from existing handlers)
        log_level = getattr(logging, level.upper(), logging.INFO)
        self.standard_logger.log(log_level, message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging performance statistics."""
        stats = self.async_handler.stats.copy()
        stats.update(self.async_handler.get_performance_stats())
        return stats
    
    def shutdown(self):
        """Shutdown async logging."""
        if self.async_handler:
            self.async_handler.stop()

# Global optimized logger
_optimized_logger = None

def get_optimized_logger(config: Optional[Dict[str, Any]] = None) -> OptimizedSystemPromptLogger:
    """Get singleton optimized logger."""
    global _optimized_logger
    if _optimized_logger is None:
        # Load config if not provided
        if config is None:
            config = load_async_logging_config()
            config = apply_environment_overrides(config)
        
        # Check if verbose logging should be enabled
        verbose_enabled = config.get("enabled", True)
        try:
            # Try to detect if we're in a performance-critical environment
            import sys
            if '--no-verbose' in sys.argv or 'pytest' in sys.modules:
                verbose_enabled = False
        except:
            pass
            
        _optimized_logger = OptimizedSystemPromptLogger(enable_verbose=verbose_enabled, config=config)
    return _optimized_logger

def log_verbose_system_prompt_phase_fast(phase: str, 
                                        message: str, 
                                        data: Any = None,
                                        correlation_id: Optional[str] = None):
    """
    Drop-in replacement for _log_verbose_system_prompt_phase with better performance.
    """
    logger = get_optimized_logger()
    logger.log_phase(phase, message, data, correlation_id)

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor performance of compression operations."""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return self.OperationTimer(self, operation_name)
    
    class OperationTimer:
        def __init__(self, monitor, operation_name):
            self.monitor = monitor
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.perf_counter() - self.start_time
                self.monitor._record_time(self.operation_name, duration)
    
    def _record_time(self, operation_name: str, duration: float):
        """Record operation timing."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        # Keep only recent measurements
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-50:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        for op_name in self.operation_times:
            times = self.operation_times[op_name]
            if times:
                stats[op_name] = {
                    'count': self.operation_counts[op_name],
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'recent_avg': sum(times[-10:]) / min(len(times), 10)
                }
        return stats

# Global performance monitor
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

# Extended async logging for all Python logging
class AsyncPythonLogHandler(logging.Handler):
    """Async handler for all Python logging calls that works alongside existing handlers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.async_handler = AsyncLogHandler(config)
        self.async_handler.start()
        self.existing_handlers = []
        
    def set_existing_handlers(self, handlers):
        """Store references to existing handlers so we can forward to them."""
        self.existing_handlers = handlers.copy()
        
    def emit(self, record):
        """Handle a logging record for async performance monitoring only."""
        try:
            # PERFORMANCE MONITORING ONLY - No logging output to avoid duplication
            # All actual logging now goes through OptimizedSystemPromptLogger or standard handlers
            
            # Only track performance metrics for relevant records
            if self._is_performance_relevant(record):
                try:
                    # Track metrics without any message output
                    self.async_handler._track_operation(
                        f"{record.name.upper()}_{record.levelname}",
                        record.created
                    )
                except Exception:
                    # Silently ignore async processing errors
                    pass
                
        except Exception:
            # Silent error handling to prevent logging loops
            pass
    
    def _is_performance_relevant(self, record):
        """Check if this record is relevant for performance monitoring."""
        # Only track specific performance-related operations
        performance_keywords = [
            'COMPRESS', 'DECOMPRESS', 'SYSTEM_PROMPT', 'METRICS', 
            'PERFORMANCE', 'ASYNC', 'BATCH', 'QUEUE'
        ]
        try:
            message = record.getMessage().upper()
            return any(keyword in message for keyword in performance_keywords)
        except:
            return False
    
    def close(self):
        """Close the async handler."""
        if self.async_handler:
            self.async_handler.stop()
        super().close()

def setup_global_async_logging(config: Optional[Dict[str, Any]] = None):
    """
    Setup async logging for all Python logging calls in the application.
    This works alongside existing handlers (like file handlers) rather than replacing them.
    
    Args:
        config: Async logging configuration dict. If None, loads from config file.
    """
    # Load config if not provided
    if config is None:
        config = load_async_logging_config()
        config = apply_environment_overrides(config)
    
    if not config.get("enabled", True):
        return
    
    # Check if we should use the new optimized approach instead
    use_optimized_only = config.get("use_optimized_logger_only", True)
    if use_optimized_only:
        # Using optimized async logger only (shown in formatted section)
        return get_optimized_logger(config)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Check if async handler is already present to prevent duplicates
    for handler in root_logger.handlers:
        if isinstance(handler, AsyncPythonLogHandler):
            async_setup_logger.info("   📝 Global async logging already enabled")
            return handler
    
    # Capture existing handlers before adding async handler
    existing_handlers = root_logger.handlers.copy()
    
    # Create async handler that works alongside existing handlers
    async_handler = AsyncPythonLogHandler(config)
    log_level = config.get("log_level", "INFO")
    async_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Configure the async handler to forward to existing handlers
    async_handler.set_existing_handlers(existing_handlers)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    async_handler.setFormatter(formatter)
    
    # SIMPLIFIED APPROACH: Add async handler alongside existing handlers, but configure it
    # to only handle performance monitoring, not replace existing logging
    
    # Add the async handler for performance monitoring only
    # Set it to a high level initially to minimize interference
    async_handler.setLevel(logging.DEBUG)  # Let it see records for performance monitoring
    
    # Add the handler last to ensure it doesn't interfere with existing setup
    root_logger.addHandler(async_handler)
    
    async_setup_logger.info(f"   📝 Added async handler alongside {len(existing_handlers)} existing handlers")
    
    # Verify handlers are working
    for i, handler in enumerate(existing_handlers):
        handler_type = type(handler).__name__
        if hasattr(handler, 'baseFilename'):
            async_setup_logger.info(f"   📄 Handler {i}: {handler_type} -> {handler.baseFilename}")
        elif hasattr(handler, 'stream'):
            stream_name = getattr(handler.stream, 'name', 'console')
            async_setup_logger.info(f"   📺 Handler {i}: {handler_type} -> {stream_name}")
        else:
            async_setup_logger.info(f"   📋 Handler {i}: {handler_type}")
    
    # Log configuration details
    queue_size = config.get("max_queue_size", 0)
    batch_size = config.get("batch_size", 50)
    worker_timeout = config.get("worker_timeout", 0.1)
    
    queue_desc = "unlimited" if queue_size == 0 else f"{queue_size:,}"
    async_setup_logger.info(f"   📝 Async logging configured: {log_level} level, {queue_desc} queue, batch={batch_size}, timeout={worker_timeout}s")
    
    if existing_handlers:
        async_setup_logger.info(f"   📝 Working with {len(existing_handlers)} existing handlers")
    
    return async_handler

def remove_global_async_logging():
    """Remove async logging handlers from root logger and restore original handlers."""
    root_logger = logging.getLogger()
    
    # Find and remove async handlers, restoring original handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, AsyncPythonLogHandler):
            # Restore original handlers if they exist
            if handler.existing_handlers:
                root_logger.handlers.clear()
                for original_handler in handler.existing_handlers:
                    root_logger.addHandler(original_handler)
                async_setup_logger.info(f"   📝 Restored {len(handler.existing_handlers)} original logging handlers")
            
            handler.close()
            if handler in root_logger.handlers:
                root_logger.removeHandler(handler)

