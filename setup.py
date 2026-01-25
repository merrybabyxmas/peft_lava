from setuptools import setup, find_packages

setup(
    name="peft",
    version="0.1.0",
    description="PEFT LAVA - Optimized for no_gate branch",
    author="merrybabyxmas",
    # 패키지가 들어있는 디렉토리를 자동으로 찾습니다.
    # ~/peft_lava/peft 폴더를 패키지로 인식하게 됩니다.
    packages=find_packages(),
    install_requires=[
        "numpy",
        "packaging",
        "psutil",
        "pyyaml",
        "tqdm",
        "datasets"
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)