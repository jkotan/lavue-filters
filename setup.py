#!/usr/bin/env python
# Copyright (C) 2017  DESY, Christoph Rosemann, Notkestr. 85, D-22607 Hamburg
#
# lavue is an image viewing program for photon science imaging detectors.
# Its usual application is as a live viewer using hidra as data source.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation in  version 2
# of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
# Author:
#     Jan Kotanski <jan.kotanski@desy.de>
#

""" setup.py for setting Lavue"""

import codecs
import os
import sys
from setuptools import setup
# from setuptools.command.build_py import build_py
# from distutils.command.clean import clean
# from distutils.util import get_platform
# import shutil

from sphinx.setup_command import BuildDoc


def read(fname):
    """ read the file

    :param fname: readme file name
    :type fname: :obj:`str`
    """
    with codecs.open(os.path.join('.', fname), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

# from sphinx.setup_command import BuildDoc


#: (:obj:`str`) package name
NAME = 'lavuefilters'
#: (:obj:`module`) package name
lavuepackage = __import__(NAME)
#: (:obj:`str`) full release version
release = lavuepackage.__version__
#: (:obj:`str`) package version
version = ".".join(release.split(".")[:2])


needs_pytest = set(['test']).intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

install_requires = [
    #    'pyqtgraph>=0.10.0',
    #    'numpy>1.6.0',
    #    'pyzmq',
    #    'scipy',
]


#: (:obj:`dict` <:obj:`str`, `any`>) metadata for distutils
SETUPDATA = dict(
    name='lavuefilters',
    version=release,
    description='Live image viewer application for photon science detectors.',
    long_description=read('README.rst'),
    install_requires=install_requires,
    url='https://github.com/jkotan/lavuefilters',
    author='J.Kotanski',
    author_email='jan.kotanski@desy.de',
    license='GPLv2',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='live viewer nexus writer plugin',
    packages=[NAME],
    # package_data=package_data,
    # include_package_data=True, # do not include image an qrc files
    zip_safe=False,
    setup_requires=pytest_runner,
    tests_require=['pytest'],
    cmdclass={
        "build_sphinx": BuildDoc
    },
    command_options={
        'build_sphinx': {
            'project': ('setup.py', NAME),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
)


def main():
    """ the main function
    """
    setup(**SETUPDATA)


if __name__ == '__main__':
    main()
