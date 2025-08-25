"""
Setup script for RE_ware: Self-Evolving Software Lifecycle Management Library
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="re_ware",
    version="0.1.0",
    author="RE_ware Development Team",
    author_email="contact@re-ware.dev",
    description="Self-evolving software lifecycle management library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/re_ware",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Project Management",
    ],
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv>=0.19.0",
        "anthropic>=0.3.0",
        "openai>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
        "graph": [
            "neo4j>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "re-ware-evolve=evolve:main",
        ],
    },
    include_package_data=True,
    package_data={
        "re_ware": ["*.md", "*.txt"],
    },
)
