echo "===== Updating the PYTHONPATH ====="
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "===== Installing dependencies ====="
poetry install

echo "===== Activating the Poetry virtual environment ====="
poetry shell
