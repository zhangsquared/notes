echo "black"
black .
echo

echo "isort"
isort .
echo

echo "pylint"
pylint ./src/ ./tests/
echo

echo "mypy"
mypy
echo
