from setuptools import setup, find_packages
from os import path

cd = path.abspath(path.dirname(__file__))

with open(path.join(cd,'README.md'), encoding='utf-8') as f:
    readme_description = f.read()

setup(
    name='crm-rfm-modeling',
    version='1.0.2',
    packages=['crm-rfm-modeling'],
    license='GNU General Public License',
    description='RFM Modeling Package for modeling Consumer behavior.',
    long_description=readme_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jzarco/RFM',
    author='Juan Zarco',
    python_requires='>=3.5',
    classifiers=[
        'Intended Audience :: Developers',
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Topic :: Software Development :: Libraries :: Python Modules'
        'Topic :: CRM Analysis :: RFM Modeling Tools',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=['pandas>=0.25.2',
                      'numpy>=1.17.3']
)