from setuptools import setup, find_packages

setup(
    name='multilevelmvmverification',
    version='1.0',
    packages=find_packages(exclude=["examples", "examples.*", "timing_analysis_and_validation", "timing_analysis_and_validation.*"]),
)
