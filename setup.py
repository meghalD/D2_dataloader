import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='detectron2',
    version='0.0.3',
    author='user',
    author_email='user@gmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/Muls/toolbox',
    project_urls = {
        "Bug Tracker": "https://github.com/Muls/toolbox/issues"
    },
    license='MIT',
    packages=['detectron2'],
    install_requires=['fvcore', 'omegaconf==2.1.0.dev22'],
)
