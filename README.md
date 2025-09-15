# EdwinBennink.github.io
This website shares notes and reflections on (medical) image analysis, programming, and mathematics.

## How to build

### Setting up a python environment
```ps
python -m venv venv
Set-ExecutionPolicy -Scope CurrentUser Unrestricted
./venv/Scripts/Activate.ps1
python -m pip install -r ./requirements.txt
```

### Rendering the static site
```ps
quarto render
```