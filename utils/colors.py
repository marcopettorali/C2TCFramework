import math
import random

import matplotlib.colors as mcolors
import numpy as np
from rich.console import Console
from rich.text import Text

# from colorist import ColorHex, ColorRGB, BgColorHex, BgColorRGB

# COLOR CONVERSIONS


def is_rgb(color):
    """
    Checks if the color is in RGB format.

    Args:
        color (tuple): A tuple representing the color in RGB format.

    Returns:
        bool: True if the color is in RGB format, False otherwise.
    """
    return type(color) == tuple and len(color) == 3


def is_hex(color):
    """
    Checks if the color is a valid hex color code.

    Args:
        color (str): The color code to check.

    Returns:
        bool: True if the color is a valid hex color code, False otherwise.
    """
    return type(color) == str and len(color) == 7 and color[0] == "#"


def is_named_color(color):
    """
    Checks if the given color is a valid named color.

    Parameters:
    color (str): The color to check.

    Returns:
    bool: True if the color is a valid named color, False otherwise.
    """

    return color in mcolors.get_named_colors_mapping()


def get_closest_named_color(color, search_space="all"):

    color = mcolors.to_rgb(color)

    closest_color = None
    closest_distance = float("inf")

    containers = {}
    if search_space == "all":
        containers = mcolors.get_named_colors_mapping()

    if "base" in search_space:
        containers.update(mcolors.BASE_COLORS)

    if "css4" in search_space:
        containers.update(mcolors.CSS4_COLORS)

    if "tab" in search_space:
        containers.update(mcolors.TABLEAU_COLORS)

    if "xkcd" in search_space:
        containers.update(mcolors.XKCD_COLORS)

    for named_color in containers:
        named_color_rgb = mcolors.to_rgb(named_color)
        distance = sum([(c1 - c2) ** 2 for c1, c2 in zip(color, named_color_rgb)])

        if distance < closest_distance:
            closest_color = named_color
            closest_distance = distance

    return closest_color


# COLOR OPERATIONS


def random_color(mode="hex"):
    """
    Generates a random color in either hex or rgb format.

    Args:
        mode (str): The format of the color to return. Can be "hex" or "rgb".

    Returns:
        str or tuple: The random color in the specified format.
    Raises:
        ValueError: If an invalid mode is provided.
    """
    color = (random.random(), random.random(), random.random())

    if mode == "hex":
        return mcolors.to_hex(color)
    elif mode == "rgb":
        return color
    else:
        raise ValueError(f"random_color(): invalid mode {mode}")


def blend(c1, c2, w1):
    """
    Blend two colors (c1 and c2) together given a weight w1 and returns the result.

    If c1 or c2 is in hexadecimal format, it will be converted to RGB format before multiplication.
    If both c1 and c2 are in hexadecimal format, the result will be returned in hexadecimal format.

    Args:
        c1 (str or tuple): The first color to be multiplied.
        c2 (str or tuple): The second color to be multiplied.
        w1 (float): The weight of c1 in the final result. Must be between 0 and 1.

    Returns:
        tuple or str: The result of multiplying c1 and c2 together. If both c1 and c2 are in hexadecimal format,
        the result will be returned in hexadecimal format. Otherwise, it will be returned in RGB format.

    Raises:
        ValueError: If the weight is not between 0 and 1.
    """

    if w1 < 0 or w1 > 1:
        raise ValueError("Weight must be between 0 and 1")

    if is_hex(c1):
        c1 = mcolors.to_rgb(c1)
    if is_hex(c2):
        c2 = mcolors.to_rgb(c2)

    val = [w1 * e1 + (1 - w1) * e2 for e1, e2 in zip(c1, c2)]

    if is_hex(c1) and is_hex(c2):
        val = mcolors.to_hex(val)

    return val


def brightness(color, amount):
    """
    Adjusts the brightness of a given color by a specified amount.

    Args:
        color (str): The color to adjust. Can be in hex or CSS4 format.
        amount (float): The amount to adjust the brightness by. Must be between -1 and 1.

    Returns:
        str: The adjusted color in the same format as the input color.
    Raises:
        ValueError: If the amount is not between -1 and 1.
    """
    if amount < -1 or amount > 1:
        raise ValueError("Amount must be between -1 and 1")

    converted = False
    if is_hex(color) or is_named_color(color):
        color = mcolors.to_rgb(color)
        converted = True

    val = [(m + amount) for m in color]

    for i in range(len(val)):
        if val[i] > 1:
            val[i] = 1

        if val[i] < 0:
            val[i] = 0

    val = tuple(val)

    if converted:
        val = mcolors.to_hex(val)

    return val


# PALETTE GENERATION


# def palette(input_colors, num_output_colors):
#     """
#     Given a list of colors and a number of desired output colors, returns a list of n colors that are evenly spaced
#     between the colors in the input list.

#     Args:
#         input_colors (list): A list of colors in any format that can be converted to RGB using the mcolors.to_rgb function.
#         num_output_colors (int): The number of output colors to generate.

#     Returns:
#         list: A list of n colors in hexadecimal format.

#     Raises:
#         ValueError: If input_colors is not a list, num_output_colors is not an integer, input_colors is empty, or num_output_colors is less than 0.
#     """

#     if num_output_colors <= len(input_colors):
#         # If the requested number of colors is less than or equal to the input colors, truncate.
#         return input_colors[:num_output_colors]

#     # Convert input colors to RGB
#     rgb_colors = [mcolors.to_rgb(color) for color in input_colors]

#     # Calculate the number of intermediate colors per segment
#     segments = len(input_colors) - 1
#     colors_per_segment = (num_output_colors - len(input_colors)) // segments
#     remainder = (num_output_colors - len(input_colors)) % segments

#     output_colors = [input_colors[0]]

#     for i in range(segments):
#         # Add interpolated colors for each segment
#         num_interpolated = colors_per_segment + (1 if i < remainder else 0)
#         for j in range(1, num_interpolated + 1):
#             t = 1-j / (num_interpolated + 1)
#             interpolated_color = blend(rgb_colors[i], rgb_colors[i + 1], t)
#             output_colors.append(mcolors.to_hex(interpolated_color))
#         output_colors.append(input_colors[i + 1])


#     return output_colors
def palette(input_color_list, num_output_colors):
    """Generates a gradient of n colors blending smoothly through the given color_list."""
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_palette", input_color_list, N=num_output_colors)
    return [mcolors.to_hex(cmap(i / (num_output_colors - 1))) for i in range(num_output_colors)]


def print_color(color, text=None, bg=True, **kwargs):
    """
    Print a color in a human-readable format and use the color to display the color.

    Args:
        color (str): The color in hexadecimal format.

    Returns:
        None
    """
    console = Console()

    hex_color = mcolors.to_hex(color)

    # Convert the color to a displayable format
    if bg:
        formatted_color = Text(f" {color if text is None else text} ", style=f"on {hex_color}")
    else:
        formatted_color = Text(f" { color if text is None else text} ", style=f"{hex_color}")

    # Display the output
    console.print(formatted_color, **kwargs)


def print_palette(palette, text=None, bg=True, **kwargs):
    """
    Print a palette in a human-readable format and use the colors to display the palette.

    Args:
        palette (list): A list of colors in hexadecimal format.

    Returns:
        None
    """
    for i, color in enumerate(palette):
        print_color(color, text=text, bg=bg, **kwargs)

    print()


def heatmap(values, color_list, lowest_to_highest=True):
    """Maps values to colors based on a gradient defined by color_list."""
    if not values:
        return []

    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return [mcolors.to_hex(mcolors.to_rgb(color_list[0]))] * len(values)

    # Normalize values between 0 and 1
    normalized = [((v - min_val) / (max_val - min_val)) for v in values]

    if not lowest_to_highest:
        normalized = [1 - n for n in normalized]

    # Generate a color gradient
    gradient = palette(color_list, 100)  # Generate a smooth gradient of 100 colors

    # Map normalized values to corresponding colors
    return [gradient[int(n * (len(gradient) - 1))] for n in normalized]


def colorize_text(text, color, bg=False):
    # rgb is a tuple of 3 values between 0 and 255
    rgb_color = mcolors.to_rgb(color)
    rgb_color = tuple([math.floor(255 * c) for c in rgb_color])

    fg_text = lambda text, color: "\33[38;2;" + f"{rgb_color[0]};{rgb_color[1]};{rgb_color[2]}" + "m" + text + "\33[0m"
    bg_text = lambda text, color: "\33[48;2;" + f"{rgb_color[0]};{rgb_color[1]};{rgb_color[2]}" + "m" + text + "\33[0m"

    if bg:
        return bg_text(text, color)
    else:
        return fg_text(text, color)


# some beautiful palette found on the internet
PALETTES = {
    # Nord Theme (https://www.nordtheme.com/docs/colors-and-palettes)
    "polar_night": ["#2e3440", "#3b4252", "#434c5e", "#4c566a"],
    "snow_storm": ["#d8dee9", "#e5e9f0", "#eceff4"],
    "frost": ["#8fbcbb", "#88c0d0", "#81a1c1", "#5e81ac"],
    "aurora": ["#bf616a", "#d08770", "#ebcb8b", "#a3be8c", "#b48ead"],
}

if __name__ == "__main__":

    print(colorize_text("Hello, world!", "green", bg=True))

    from rich import print as rprint
    from rich.markup import escape

    rprint(escape(colorize_text("Hello, world!", "green", bg=True)))

    pass
