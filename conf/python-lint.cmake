# flake8 + black --diff
add_custom_target(check-python-format)
add_custom_command(
    TARGET check-python-format PRE_BUILD
    # all *.py files
    COMMAND echo flake8
    COMMAND python -m flake8 "${CMAKE_SOURCE_DIR}"
    COMMAND echo black
    COMMAND python -m black --diff "${CMAKE_SOURCE_DIR}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# black
add_custom_target(apply-python-format)
add_custom_command(
    TARGET apply-python-format PRE_BUILD
    # all *.py files
    COMMAND python -m black "${CMAKE_SOURCE_DIR}"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
