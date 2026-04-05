import sys
path = r"c:\Users\varun\FindMySpot\database_setup.sql"
with open(path) as f:
    lines = f.readlines()
for idx,line in enumerate(lines, start=1):
    if 130 <= idx <= 170:
        sys.stdout.write(f"{idx:3d} {line}")
