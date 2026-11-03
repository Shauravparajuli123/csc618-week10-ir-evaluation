"""Download the Lucene 10.5.0 jar bundled with Pyserini 2.4.0 from PyPI into lib/."""
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

JAR_SUFFIX = "-fatjar.jar"
lib = pathlib.Path("lib")
lib.mkdir(exist_ok=True)

if list(lib.glob(f"*{JAR_SUFFIX}")):
    print("Lucene jar already present")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "pyserini==2.4.0", "--no-deps", "-d", tmp, "-q"],
        check=True,
    )
    package = next(pathlib.Path(tmp).iterdir())
    if package.suffix == ".whl":
        with zipfile.ZipFile(package) as archive:
            name = next(n for n in archive.namelist() if n.endswith(JAR_SUFFIX))
            with archive.open(name) as src, open(lib / pathlib.Path(name).name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(package) as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith(JAR_SUFFIX))
            with archive.extractfile(member) as src:
                with open(lib / pathlib.Path(member.name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
print("Lucene jar saved to lib/")
