from setuptools import setup, find_packages
import politicianmap

setup(name=politicianmap.__name__,
      version=politicianmap.__version__,
      url='https://github.com/lovit/politicianmap',
      author=politicianmap.__author__,
      author_email='soy.lovit@gmail.com',
      description='Analysis of Korean politician with news and user-generated comments',
      packages=find_packages(),
      long_description=open('README.md').read(),
      zip_safe=False,
      setup_requires=['soynlp>=0.0.4']
)