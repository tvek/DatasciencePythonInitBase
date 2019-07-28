with open('requirements.txt') as f:
    requirements = f.read().splitlines()

from setuptools import setup

setup(
    name='TestDNN',
    version='1.0',
    install_requires=requirements,
    packages=['src', 'src.test', 'src.logger', 'src.TVEKDNN'],
    url='',
    license='',
    author='Thomas Vimal Easo K',
    author_email='thomasvml@gmail.com',
    description=''
)
