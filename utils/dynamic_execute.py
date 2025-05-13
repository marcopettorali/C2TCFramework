import importlib
import re

def dynamic_execute(call_string, *additional_args, **additional_kwargs):
    """
    Dynamically loads a module, extracts function and arguments from `call_string`,
    appends additional arguments, and executes the function.

    :param call_string: Function call string (e.g., 'A.B.script.sum(3,5,method="standard")')
    :param additional_args: Extra positional arguments to append
    :param additional_kwargs: Extra keyword arguments to append
    :return: The result of the function execution
    """
    
    # Extract function and arguments using regex
    match = re.match(r"([\w\.]+)\((.*)\)", call_string)
    if not match:
        raise ValueError(f"Invalid function call format: {call_string}")
    
    func_path, args_str = match.groups()

    # Split module path and function name
    *module_path, function_name = func_path.split(".")
    module_name = ".".join(module_path)

    # Initialize args and kwargs
    args, kwargs = [], {}

    # If arguments exist, process them
    if args_str:
        # Split positional arguments and keyword arguments correctly
        args_list = args_str.split(",")

        for arg in args_list:
            arg = arg.strip()
            if "=" in arg:  # Keyword argument
                key, value = arg.split("=", 1)
                key = key.strip()
                value = eval(value.strip())  # Safely evaluate value
                kwargs[key] = value
            else:  # Positional argument
                args.append(eval(arg))  # Safely evaluate positional args

    # Append additional args and kwargs
    args.extend(additional_args)
    kwargs.update(additional_kwargs)

    # Dynamically import module
    module = importlib.import_module(module_name)

    # Retrieve function
    function = getattr(module, function_name)

    # Execute function with combined arguments
    return function(*args, **kwargs)

