#!/usr/bin/env python3
import importlib.util
from distutils.command.build import build as DistutilsBuild
from os.path import abspath, join, dirname, realpath
from setuptools import find_packages, setup

with open(join("gym2", "version.py")) as version_file:
    exec(version_file.read())


class Build(DistutilsBuild):
    def run(self):
        # Pre-compile the Cython
        current_path = abspath(dirname(__file__))
        builder_path = join(current_path, 'gym2', 'builder.py')
        spec = importlib.util.spec_from_file_location(
            "gym2.builder", builder_path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)

        DistutilsBuild.run(self)


def read_requirements_file(filename):
    req_file_path = '%s/%s' % (dirname(realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'gym2' or p.startswith('gym2.')


NAME = 'gym2'
DESCRIPTION = 'gym2: gym harder'
URL = 'https://github.com/vlad17/gym2'
EMAIL = 'vladf@berkeley.edu'
AUTHOR = 'Vladimir Feinberg'

setup(
    name=NAME,
    version=__version__,
    description=DESCRIPTION,
    long_description='',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    include_package_data=True,
    license='Apache 2.0',
    packages=packages,
    package_dir={'gym2': 'gym2'},
    # hacky, should be built once when packaging
    # this builds on every install
    package_data={'gym2': ['mujoco/generated/*.so',
                           '*.pyx', 'mujoco/*.pyx',
                           'mujoco/pxd/*.pyx', 'mujoco/pxd/*.pxd',
                           'mujoco/gl/*.h', 'mujoco/gl/*.c',
                           'mujoco/generated/*.pxi']},
    install_requires=read_requirements_file('requirements.txt'),
    tests_require=read_requirements_file('requirements.dev.txt'),
    # Add requirements for builder.py here since there's no
    # guarantee that they've been installed before this setup script
    # is run. (The install requirements only guarantee that those packages
    # are installed as part of installation. No promises about order.)
    setup_requires=read_requirements_file('requirements.txt'),
    cmdclass={'build': Build},
)
