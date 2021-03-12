echo "Updating the PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Installing dependancies"
poetry install

echo "Activating the Poetry vitrual environment"
poetry shell
