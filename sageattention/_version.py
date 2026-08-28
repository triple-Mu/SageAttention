# Single source of truth for the package version. Keep this file free of
# imports: setuptools reads it statically (attr:) during isolated builds where
# torch may be unavailable, and CMake greps it for SAGE_VERSION.
__version__ = "2.2.0"
