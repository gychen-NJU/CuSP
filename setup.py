from setuptools import setup, find_packages

setup(
    name='CuSP',
    version='0.1.0',
    description='Cuda-supported SpectroPolarimetry',
    author='Chen Guoyin',
    author_email='gychen@smail.nju.edu.cn',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={'CuSP': ['examples/*.ipynb']},
    include_package_data=True,
    install_requires=[
        'torch>=1.13.1',
        'numpy>=1.24.3',
        'matplotlib>=3.7.2',
        'scipy>=1.11.3',
    ],
)
