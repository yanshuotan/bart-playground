from setuptools import setup, find_packages

def parse_requirements(file_path):
    with open(file_path, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith("#")
        ]
    return requirements

setup(
    name="bart_playground",          
    version="0.1.0",        
    packages=["bart_playground"],          
    install_requires=parse_requirements("requirements.txt"),              
    python_requires=">=3.7",            
    description="A fast and modular implementation of BART",
    author="Yan Shuo Tan",
    author_email="yanshuo@nus.edu.sg",
    url="https://github.com/yanshuotan/bart-playground", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
