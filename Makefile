# ------------------------------------------------------------
# Prepare
# ------------------------------------------------------------
# Directories to check
check_dirs := src

# Test the local checkout rather than the installed
export PYTHONPATH = src

# ------------------------------------------------------------
# Format code
# ------------------------------------------------------------
format-check:
	ruff check $(check_dirs)
	ruff format --diff $(check_dirs)

format-fixup:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)
