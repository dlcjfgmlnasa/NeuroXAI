# -*- coding:utf-8 -*-
from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup


setup(name='neuroxai',
      version='0.0.0.1',
      description='NeuroXAI: Adaptive, robust, explainable surrogate framework for determination of channel importance '
                  'in EEG application',
      url='https://github.com/dlcjfgmlnasa/NeuroXAI',
      author='Cheol-Hui Lee',
      long_description=open('README.md', 'r').read(),
      long_description_content_type='text/markdown',
      author_email='dlcjfgmlnasa28@korea.ac.kr',
      license='Apache License 2.0',
      packages=find_packages(where='src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
      python_requires='>=3.5',
      install_requires=[
          'lime',
          'mne >= 1.8.0',
      ],
      include_package_data=True,
      zip_safe=False)
