import setuptools

with open(r'README.md', mode=r'r') as readme_handle:
    long_description = readme_handle.read()

setuptools.setup(
    name=r'hopfield-layers-dahn',
    version=r'0.1.0',
    author=r'Priyam Ghosh',
    author_email=r'',
    url=r'https://github.com/hopfileds/hopfield-layers',
    description=r'DAHN: Diffusion-Augmented Hopfield Networks (fork of ml-jku/hopfield-layers)',
    long_description=long_description,
    long_description_content_type=r'text/markdown',
    packages=setuptools.find_packages(),
    python_requires=r'>=3.8.0',
    install_requires=[
        r'torch>=1.5.0',
        r'numpy>=1.20.0'
    ],
    zip_safe=True
)
