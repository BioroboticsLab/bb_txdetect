#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]


setup(
    name='bb_txdetect',
    version='0.1',
    description='BeesBook trophallaxis detection',
    author='Mathis Hocke',
    author_email='mathis.hocke@fu-berlin.de',
    url='https://github.com/BioroboticsLab/bb_txdetect/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['bb_txdetect'],
    package_dir={'bb_txdetect': 'bb_txdetect/'}
)
