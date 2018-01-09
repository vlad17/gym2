from setuptools import setup

NAME = 'gym2'
DESCRIPTION = 'gym2: gym harder'
URL = 'https://github.com/vlad17/gym2'
EMAIL = 'vladf@berkeley.edu'
AUTHOR = 'Vladimir Feinberg'

# What packages are required for this module to be executed?
REQUIRED = [
    # too lazy to figure this out yet
]

setup(
    name=NAME,
    version='0.0.1',
    description=DESCRIPTION,
    long_description='',
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    install_requires=REQUIRED,
    include_package_data=True,
    license='Apache 2.0'
)
