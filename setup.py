from setuptools import setup, find_packages

setup(
    name='SentenceVAE-Transformers',
    version='0.1.0',
    author='Eviatar Nachshoni, Itay Etelis',
    author_email='nachshoni.eviatar@gmail.com, etelis.itay@gmail.com',
    description='Enhancing Sentence Generation with Transformer Models. This package implements ideas inspired by Samuel R. Bowman et al.\'s paper on using Variational Autoencoders for sentence generation in NLP.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Etelis/SentenceVAE-Transformers',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'matplotlib',
        'tensorboardX'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',
)
