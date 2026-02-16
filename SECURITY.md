# Security Policy

## Supported Versions

Currently supported versions for security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| 0.3.x   | :x:                |
| 0.2.x   | :x:                |
| 0.1.x   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in compact-tree, please report it privately:

1. **Do not** open a public GitHub issue
2. Email the maintainers at: your.email@example.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and provide a timeline for a fix.

## Security Considerations

This library:
- Is designed for **read-only** access to data structures
- Does not execute arbitrary code from serialized files
- Uses standard Python serialization (struct, bitarray)
- Supports `pickle` for convenience; **only unpickle data from trusted sources**
- Supports fsspec for file access (be cautious with remote URLs)

When using `CompactTree` with remote storage:
- Only load files from trusted sources
- Be aware of potential path traversal issues with user-supplied URLs
- Consider implementing URL validation in your application
