# Joyl Language Changelog

## [Unreleased]
### Added
- Support for pattern matching with `match` expressions
- New built-in functions: `zip()`, `enumerate()`
### Changed
- Improved error messages for type mismatches
### Fixed
- Memory leak in recursive function calls (#127)

## [1.2.0] - 2024-03-15
### Breaking Changes
- `let` now requires type annotations in strict mode
### Added
- Experimental WASM compilation target
- `with` statement for resource management
### Deprecated
- Old string interpolation syntax (`%` operator)

## [1.1.4] - 2024-02-28
### Fixed
- Parser crash on nested ternary operations (#115)
- Incorrect scoping of loop variables (#118)

## [1.1.0] - 2024-01-10
### Added
- Null coalescing operator (`??`)
- Standard library modules:
  - `json`
  - `datetime`
### Performance
- 40% faster array operations
- Reduced memory usage by 15%

## [1.0.0] - 2023-11-05
### Initial Release
- Core language features:
  - Variables and functions
  - Control flow
  - Classes and traits
- Virtual machine implementation
- Basic standard library

---

### Versioning Scheme
We follow [Semantic Versioning](https://semver.org/):
- `MAJOR`: Breaking changes
- `MINOR`: New features
- `PATCH`: Bug fixes

### Viewing Detailed Changes
Full commit history available on [GitHub](https://github.com/vd437/joyl/commits/main)
