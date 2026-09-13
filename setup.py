"""Setuptools compatibility configuration for installing launcher scripts."""

from setuptools import setup


setup(
    scripts=["codex-ephemeral", "ephemeral-agent", "ephemeral-session-env"],
)
