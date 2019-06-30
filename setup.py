from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='TestDNN',
    version='',
    packages=['src', 'src.test', 'src.logger', 'src.TVEKDNN'],
    url='',
    license='',
    author='Thomas Vimal Easo K',
    author_email='thomasvml@gmail.com',
    description='',
    install_requires=requirements,
)
