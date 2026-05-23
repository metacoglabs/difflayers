import setuptools

with open(r'README.md', mode=r'r') as readme_handle:
    long_description = readme_handle.read()

setuptools.setup(
    name=r'difflayers',
    version=r'0.1.0',
    author=r'Priyam Ghosh',
    author_email=r'',
    url=r'https://github.com/hopfileds/hopfield-layers',
    description=r'difflayers: Diffusion-Augmented Hopfield Networks',

    long_description=long_description,
    long_description_content_type=r'text/markdown',
    license=r'BSD',
    packages=setuptools.find_packages(exclude=['examples*', 'notebooks*', 'data*',
                                               'results*', 'src*', 'bench*']),
    python_requires=r'>=3.8',
    install_requires=[
        r'torch>=1.9.0',
        r'numpy>=1.20.0',
        r'scipy>=1.7.0',
    ],
    classifiers=[
        r'Development Status :: 3 - Alpha',
        r'Intended Audience :: Science/Research',
        r'License :: OSI Approved :: BSD License',
        r'Programming Language :: Python :: 3',
        r'Programming Language :: Python :: 3.8',
        r'Programming Language :: Python :: 3.9',
        r'Programming Language :: Python :: 3.10',
        r'Programming Language :: Python :: 3.11',
        r'Programming Language :: Python :: 3.12',
        r'Topic :: Scientific/Engineering :: Artificial Intelligence',
        r'Operating System :: OS Independent',
    ],
    keywords=r'hopfield networks deep learning attention diffusion graph',
    zip_safe=False,
)
