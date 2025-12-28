import os
import sys
import shutil
import subprocess
from setuptools import setup
from setuptools.command.build import build

class BuildZig(build):
    def run(self):
        build.run(self)
        subprocess.check_call([sys.executable, '-m', 'ziglang', 'build', 'lib'])
        source_dir = os.path.abspath('zig-out')
        lib_dir = os.path.join(self.build_lib, 'zig-out')
        if os.path.exists(lib_dir):
            shutil.rmtree(lib_dir)
        shutil.copytree(source_dir, lib_dir)

setup(
    name="tokenizerz",
    version="0.0.3a5",
    author="J Joe",
    author_email="backupjjoe@gmail.com",
    description="Minimal BPE tokenizer in Zig",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jaco-bro/tokenizer",
    py_modules=['tokenizerz'],
    # python_requires=">=3.12.8",
    # install_requires=['ziglang==0.13.0.post1', 'jinja2==3.1.5'],
    install_requires=['ziglang==0.13.0.post1', 'jinja2'],
    cmdclass={ 'build': BuildZig },
    data_files=[
        ('', ['build.zig', 'build.zig.zon']),
        ('src', ['src/tokenizer.zig']),
    ],
    entry_points={"console_scripts": ["bpe=tokenizerz:run"]},
    zip_safe=False,
)
