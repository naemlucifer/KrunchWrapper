#!/usr/bin/env python3
"""
Test compression through KrunchWrapper API and generate output files.
This script demonstrates the complete compression pipeline:
1. Original content (before.txt)
2. Compressed content (compressed.txt)  
3. System prompt with decompression instructions (system_prompt.txt)
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.compress import compress_with_dynamic_analysis, decompress
from core.system_prompt import build_system_prompt
from core.model_context import set_global_model_context


def create_sample_content():
    """Create a realistic sample that will benefit from compression."""
    return """
import logging
import os
import sys
import json
import pathlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import aiohttp
from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, Field

# Configuration and logging setup
logger = logging.getLogger(__name__)
config_path = pathlib.Path(__file__).parent / "config" / "config.json"

class DataProcessor:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.configuration = self.load_configuration()
        self.data_cache = {}
        
    def load_configuration(self) -> Dict[str, Any]:
        \"\"\"Load configuration from JSON file.\"\"\"
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                configuration = json.load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return configuration
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {self.config_path}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            return {}
            
    def process_data_async(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        \"\"\"Process data asynchronously with error handling.\"\"\"
        processed_data = []
        for item in data:
            try:
                processed_item = self.transform_data_item(item)
                if self.validate_data_item(processed_item):
                    processed_data.append(processed_item)
                    self.logger.debug(f"Successfully processed data item: {item.get('id', 'unknown')}")
                else:
                    self.logger.warning(f"Validation failed for data item: {item}")
            except Exception as e:
                self.logger.error(f"Error processing data item {item}: {e}")
                continue
        return processed_data
        
    def transform_data_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Transform a single data item according to configuration rules.\"\"\"
        transformation_rules = self.configuration.get("transformation_rules", {})
        transformed_item = item.copy()
        
        for field_name, transformation_rule in transformation_rules.items():
            if field_name in transformed_item:
                original_value = transformed_item[field_name]
                if transformation_rule["type"] == "uppercase":
                    transformed_item[field_name] = original_value.upper()
                elif transformation_rule["type"] == "lowercase":
                    transformed_item[field_name] = original_value.lower()
                elif transformation_rule["type"] == "prefix":
                    prefix_value = transformation_rule.get("prefix", "")
                    transformed_item[field_name] = f"{prefix_value}{original_value}"
                    
        return transformed_item
        
    def validate_data_item(self, item: Dict[str, Any]) -> bool:
        \"\"\"Validate a data item against configuration schema.\"\"\"
        validation_rules = self.configuration.get("validation_rules", {})
        
        for field_name, validation_rule in validation_rules.items():
            if validation_rule.get("required", False) and field_name not in item:
                self.logger.error(f"Required field missing: {field_name}")
                return False
                
            if field_name in item:
                field_value = item[field_name]
                if "min_length" in validation_rule:
                    if len(str(field_value)) < validation_rule["min_length"]:
                        self.logger.error(f"Field {field_name} too short: {len(str(field_value))}")
                        return False
                        
                if "max_length" in validation_rule:
                    if len(str(field_value)) > validation_rule["max_length"]:
                        self.logger.error(f"Field {field_name} too long: {len(str(field_value))}")
                        return False
                        
        return True
        
    async def fetch_external_data(self, endpoint_url: str) -> Optional[Dict[str, Any]]:
        \"\"\"Fetch data from external API endpoint.\"\"\"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(endpoint_url) as response:
                    if response.status == 200:
                        external_data = await response.json()
                        self.logger.info(f"Successfully fetched data from {endpoint_url}")
                        return external_data
                    else:
                        self.logger.error(f"HTTP {response.status} error fetching from {endpoint_url}")
                        return None
        except aiohttp.ClientError as e:
            self.logger.error(f"Network error fetching from {endpoint_url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error fetching from {endpoint_url}: {e}")
            return None
            
    def save_processed_data(self, processed_data: List[Dict[str, Any]], output_path: str):
        \"\"\"Save processed data to JSON file.\"\"\"
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Processed data saved to {output_path}")
        except IOError as e:
            self.logger.error(f"Error saving processed data to {output_path}: {e}")
            raise

class APIHandler:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app = FastAPI(title="Data Processing API")
        self.setup_routes()
        
    def setup_routes(self):
        \"\"\"Setup FastAPI routes for the data processing API.\"\"\"
        
        @self.app.post("/process")
        async def process_data_endpoint(request: Request):
            \"\"\"Process data through the data processing pipeline.\"\"\"
            try:
                request_data = await request.json()
                input_data = request_data.get("data", [])
                
                if not isinstance(input_data, list):
                    raise HTTPException(status_code=400, detail="Data must be a list")
                    
                processed_data = self.data_processor.process_data_async(input_data)
                
                return {
                    "status": "success",
                    "processed_items": len(processed_data),
                    "data": processed_data
                }
            except Exception as e:
                self.logger.error(f"Error in process_data_endpoint: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/health")
        async def health_check():
            \"\"\"Health check endpoint for monitoring.\"\"\"
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

def main():
    \"\"\"Main application entry point.\"\"\"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting data processing application")
    
    try:
        # Initialize data processor
        config_file = "config/settings.json"
        data_processor = DataProcessor(config_file)
        
        # Initialize API handler
        api_handler = APIHandler(data_processor)
        
        logger.info("Application initialized successfully")
        
        # Example usage
        sample_data = [
            {"id": 1, "name": "example_item", "value": 100},
            {"id": 2, "name": "another_item", "value": 200},
            {"id": 3, "name": "third_item", "value": 300}
        ]
        
        processed_results = data_processor.process_data_async(sample_data)
        logger.info(f"Processed {len(processed_results)} items successfully")
        
        # Save results
        output_file = "output/processed_data.json"
        data_processor.save_processed_data(processed_results, output_file)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""


def run_compression_test():
    """Run the complete compression test and generate output files."""
    print("🚀 Running KrunchWrapper Compression Test")
    print("=" * 60)
    
    # Set up model context
    set_global_model_context("gpt-4")
    
    # Generate timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create sample content
    original_content = create_sample_content()
    
    print(f"📝 Original content: {len(original_content):,} characters")
    print(f"📊 Content preview: {original_content[:200]}...")
    
    # Run compression
    print("\n🗜️ Running compression...")
    start_time = time.time()
    
    try:
        packed = compress_with_dynamic_analysis(
            original_content, 
            skip_tool_detection=False, 
            cline_mode=False
        )
        
        compression_time = time.time() - start_time
        
        # Calculate compression metrics
        original_size = len(original_content)
        compressed_size = len(packed.text)
        compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0
        
        print(f"✅ Compression completed in {compression_time:.3f}s")
        print(f"📊 Compressed size: {compressed_size:,} characters")
        print(f"📊 Compression ratio: {compression_ratio:.1%}")
        print(f"📊 Rules generated: {len(packed.used)}")
        
        # Generate system prompt
        print("\n🗣️ Generating system prompt...")
        system_prompt, metadata = build_system_prompt(
            used=packed.used,
            lang="python",
            format_name="chatml",
            user_content="Please analyze this Python code and explain its functionality.",
            cline_mode=False
        )
        
        print(f"✅ System prompt generated ({len(system_prompt):,} characters)")
        
        # Test decompression
        print("\n🔄 Testing decompression...")
        decompressed = decompress(packed.text, packed.used)
        roundtrip_success = decompressed == original_content
        
        print(f"✅ Decompression {'successful' if roundtrip_success else 'failed'}")
        
        # Create output directory
        output_dir = Path("compression_test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Save files
        print(f"\n💾 Saving output files to {output_dir}/...")
        
        # 1. Original content (before)
        before_file = output_dir / f"before_{timestamp}.txt"
        with open(before_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"   📄 Before: {before_file}")
        
        # 2. Compressed content
        compressed_file = output_dir / f"compressed_{timestamp}.txt"
        with open(compressed_file, 'w', encoding='utf-8') as f:
            f.write(packed.text)
        print(f"   🗜️  Compressed: {compressed_file}")
        
        # 3. System prompt
        system_prompt_file = output_dir / f"system_prompt_{timestamp}.txt"
        with open(system_prompt_file, 'w', encoding='utf-8') as f:
            f.write(system_prompt)
        print(f"   🗣️  System Prompt: {system_prompt_file}")
        
        # 4. Compression dictionary (bonus)
        dictionary_file = output_dir / f"dictionary_{timestamp}.json"
        with open(dictionary_file, 'w', encoding='utf-8') as f:
            json.dump(packed.used, f, indent=2, ensure_ascii=False)
        print(f"   📚 Dictionary: {dictionary_file}")
        
        # 5. Metadata and stats (bonus)
        stats_file = output_dir / f"stats_{timestamp}.json"
        stats = {
            "timestamp": timestamp,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "rules_count": len(packed.used),
            "compression_time_seconds": compression_time,
            "roundtrip_success": roundtrip_success,
            "language": "python",
            "system_prompt_size": len(system_prompt),
            "metadata": metadata,
            "top_rules": dict(list(packed.used.items())[:10]) if packed.used else {}
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"   📊 Stats: {stats_file}")
        
        # Print summary
        print(f"\n📋 COMPRESSION TEST SUMMARY")
        print("=" * 60)
        print(f"✅ Test completed successfully!")
        print(f"📏 Original size: {original_size:,} characters")
        print(f"🗜️  Compressed size: {compressed_size:,} characters")
        print(f"📊 Compression ratio: {compression_ratio:.1%}")
        print(f"💾 Saved {compression_ratio*100:.1f}% space")
        print(f"📚 Dictionary entries: {len(packed.used)}")
        print(f"⏱️  Processing time: {compression_time:.3f}s")
        print(f"🔄 Roundtrip test: {'✅ PASSED' if roundtrip_success else '❌ FAILED'}")
        print(f"📁 Output directory: {output_dir.absolute()}")
        
        if packed.used:
            print(f"\n🎯 Top compression rules:")
            for i, (symbol, pattern) in enumerate(list(packed.used.items())[:10]):
                savings = (len(pattern) - len(symbol)) * packed.text.count(symbol)
                print(f"   {i+1:2d}. '{pattern}' → '{symbol}' (saves ~{savings} chars)")
        
        return True
        
    except Exception as e:
        print(f"❌ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_compression_test()
    sys.exit(0 if success else 1) 