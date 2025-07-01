# Changelog

## [Unreleased]

### Added
- Token-level compression system that evaluates each substitution independently
- Multithreading support for compression with configurable thread count
- Advanced configuration options in `config/config.jsonc`
- Aggressive compression mode for large files
- `test_token_compression.py` script for comparing compression methods
- Improved token estimation for more accurate compression decisions

### Changed
- Updated API server to use token-level compression when configured
- Enhanced README with documentation for new compression features
- Improved config documentation with new options

## [1.0.0] - Initial Release

### Added
- Basic compression system using language-specific dictionaries
- Language detection for automatic dictionary selection
- OpenAI-compatible API proxy with compression/decompression
- Support for multiple programming languages 