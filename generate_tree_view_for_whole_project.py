import os

def list_files(startpath, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = ["venv", "__pycache__", ".venv"
            , ".git", ".idea"]

    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

list_files(".")
