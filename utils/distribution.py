import json
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from rich.console import Console, Theme
from scipy.interpolate import interp1d

# set up console for rich
custom_theme = Theme({"info": "dim cyan", "warning": "bold yellow", "danger": "bold red"})
console = Console(theme=custom_theme)

print = console.print


class Distribution:
    """
    A class to represent and manipulate probability distributions.

    This class provides methods for creating, normalizing, scaling, combining,
    and convolving distributions. It also supports integration, computing
    percentiles, and representing the distribution as a string for debugging
    and visualization.

    Attributes:
        PRECISION (float): Step size for interpolation and convolution.
        LIMIT_LABEL_LENGTH (int): Maximum length of labels for distributions.

    Methods:
        __init__(xs, ys, label=None): Initializes the distribution with given x and y values.
        data: Returns the xs and ys of the distribution for plotting or processing.
        normalize(): Normalizes the distribution to ensure the total integral equals 1.
        dirac_delta(value): Creates a Dirac delta distribution at a given value.
        from_dataset(data, bins=100, range=None): Creates a distribution from a dataset.
        from_file(filename): Loads a distribution from a JSON file.
        _convolve(other): Performs convolution with another distribution.
        __add__(other): Adds (convolves) two distributions.
        percentile(p): Calculates the p-th percentile of the distribution.
        quantile(q): Calculates the q-th quantile of the distribution.
        scale(factor): Scales the distribution's spread by a given factor.
        combine(distributions, weights): Combines multiple distributions using weights.
        integrate(a=-np.inf, b=np.inf): Integrates the distribution over a range [a, b].
        mean_value(): Computes the mean value of the distribution.
        __repr__(): Returns a string representation of the distribution.
        __str__(): Returns the string representation of the distribution.
    """

    PRECISION = 0.1
    LIMIT_LABEL_LENGTH = 20

    def __init__(self, xs, ys, label=None):
        """
        Initializes the distribution with given x (xs) and y (ys) values.

        Args:
            xs (list or np.ndarray): The x-coordinates of the distribution.
            ys (list or np.ndarray): The y-coordinates of the distribution.
            label (str, optional): Label for the distribution. Default is None.
        """
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length.")

        # ensure xs are equally spaced. Otherwise sample
        if len(xs) > 1:
            first_delta = xs[1] - xs[0]
            if not np.isclose(np.diff(xs), first_delta).all():
                new_xs = np.arange(xs[0], xs[-1], Distribution.PRECISION)
                interp_func = interp1d(xs, ys, bounds_error=False, fill_value=0)

                ys = interp_func(new_xs)
                xs = new_xs

        # put ys where np.isclose(ys, 0) to 0
        ys[np.isclose(ys, 0, atol=0.0001)] = 0

        # if there are negative values for ys, raise a warning
        if (ys < -0.0001).any():
            raise ValueError(f"Negative values in the distribution are not supported: {ys[ys < 0]}")

        self._xs = xs
        self._ys = ys
        self.label = label

    @property
    def data(self):
        """
        Returns the x and y values of the distribution, handling single-value cases for plotting.

        Returns:
            tuple: A tuple of x and y values.
        """
        if len(self._xs) == 3:
            return [self._xs[1] + x for x in [-Distribution.PRECISION, 0, Distribution.PRECISION]], self._ys
        return self._xs, self._ys

    def _get_data_size(self):
        return len(self._xs)

    def normalize(self):
        """
        Normalizes the distribution to ensure the total integral equals 1.

        Returns:
            Distribution: The normalized distribution.
        """
        integral = np.trapezoid(self._ys, self._xs)
        if not np.isclose(integral, 1) and integral > 0:
            self._ys /= integral

        return self

    def clean_zeros(self):
        """
        Removes zero values from the beginning and end of the distribution.
        """

        # remove zeros from the beginning and end
        while self._ys[0] == 0:
            self._xs = self._xs[1:]
            self._ys = self._ys[1:]

        while self._ys[-1] == 0:
            self._xs = self._xs[:-1]
            self._ys = self._ys[:-1]

        return self

    @staticmethod
    def dirac_delta(value):
        """
        Creates a Dirac delta distribution centered at a specific value.

        Args:
            value (float): The center of the Dirac delta.

        Returns:
            Distribution: The Dirac delta distribution.
        """
        return Distribution.from_dataset([value])

    @staticmethod
    def from_dataset(data, bins=100, range=None):
        """
        Creates a distribution from a dataset by calculating a histogram.

        Args:
            data (list): The dataset to create the distribution from.
            bins (int, optional): Number of bins for the histogram. Default is 100.
            range (tuple, optional): Range for the histogram. Default is None.

        Returns:
            Distribution: The resulting distribution.
        """
        if len(data) == 1:
            bins = 3

        histogram, bin_edges = np.histogram(data, bins=bins, range=range, density=True)
        bin_edges = (bin_edges[1:] + bin_edges[:-1]) / 2

        return Distribution(bin_edges, histogram).normalize()

    @staticmethod
    def from_file(filename, bins=100):
        """
        Loads a distribution from a JSON file.

        Args:
            filename (str): The path to the JSON file.

        Returns:
            Distribution: The distribution loaded from the file.
        """
        if filename.endswith(".json"):
            with open(filename, "r") as file:
                data = json.load(file)
                return Distribution.from_dataset(data, bins=bins)
        else:
            raise ValueError(f"File extension not supported: {filename}")

    def _convolve(self, other):
        """
        Performs convolution with another distribution.

        Args:
            other (Distribution): The other distribution to convolve with.

        Returns:
            Distribution: The resulting convolved distribution.
        """
        xs1, ys1 = self._xs, self._ys
        xs2, ys2 = other._xs, other._ys

        min_x = min(xs1[0], xs2[0])
        max_x = max(xs1[-1], xs2[-1])

        common_xs = np.arange(min_x, max_x, Distribution.PRECISION)

        interp_func1 = interp1d(xs1, ys1, bounds_error=False, fill_value=0)
        interp_func2 = interp1d(xs2, ys2, bounds_error=False, fill_value=0)
        common_ys1 = interp_func1(common_xs)
        common_ys2 = interp_func2(common_xs)

        ys_conv = np.convolve(common_ys1, common_ys2, mode="full") * Distribution.PRECISION

        xs_conv = np.linspace(common_xs[0] + common_xs[0], common_xs[-1] + common_xs[-1], len(ys_conv))

        label = f"Convolved_"
        if self.label:
            label += f"{self.label.replace('Convolved', '')}"
        if other.label:
            label += f"{other.label.replace('Convolved', '')}"

        label = label.strip("_")

        return Distribution(xs_conv, ys_conv, label=label).normalize().clean_zeros()

    def __add__(self, other):
        """
        Adds (convolves) this distribution with another.

        Args:
            other (Distribution): The other distribution to add.

        Returns:
            Distribution: The resulting convolved distribution.
        """
        if not isinstance(other, Distribution):
            return NotImplemented
        return self._convolve(other)

    def percentile(self, p):
        """
        Calculates the p-th percentile of the distribution.

        Args:
            p (float): The percentile to calculate (0-100).

        Returns:
            float: The value at the given percentile.
        """
        if p < 0 or p > 100:
            raise ValueError("p must be in the range [0, 100].")
        if p < 1:
            console.print(
                f"Warning! percentile() expects p in the range [0, 100]. Got p={p}. If you want to use p in the range [0, 1], use quantile() instead.",
                style="warning",
            )
        return np.percentile(self._xs, p, method="inverted_cdf", weights=self._ys)

    def quantile(self, q):
        """
        Calculates the q-th quantile of the distribution.

        Args:
            q (float): The quantile to calculate (0-1).

        Returns:
            float: The value at the given quantile.
        """
        if q < 0 or q > 1:
            raise ValueError("q must be in the range [0, 1]. If you want to use q in the range [0, 100], use percentile() instead.")

        return self.percentile(q * 100)

    def scale(self, factor):
        """
        Scales the distribution's spread by a given factor.

        Args:
            factor (float): The scale factor. A larger factor spreads the distribution,
                            while a smaller factor concentrates it.

        Returns:
            Distribution: The scaled distribution.
        """
        scaled_xs = self._xs * factor
        scaled_ys = self._ys / factor

        label = f"Scaled_{self.label}" if self.label else "Scaled"
        return Distribution(scaled_xs, scaled_ys, label=label).normalize()

    def __mul__(self, factor):
        """
        Scales the distribution by multiplication with a factor.

        Args:
            factor (float): The scale factor.

        Returns:
            Distribution: The scaled distribution.
        """
        return self.scale(factor)

    __rmul__ = __mul__

    @staticmethod
    def combine(distributions, weights):
        """
        Combines multiple distributions using the total probability rule.

        Args:
            distributions (list): List of distributions to combine.
            weights (list): List of weights for each distribution. Must sum to 1.

        Returns:
            Distribution: The combined distribution.
        """
        if len(distributions) != len(weights):
            raise ValueError("Number of distributions and weights must match.")

        weights = np.array(weights)
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Weights must sum to 1.")

        min_x = min(d._xs[0] for d in distributions)
        max_x = max(d._xs[-1] for d in distributions)
        common_xs = np.arange(min_x, max_x, Distribution.PRECISION)

        combined_ys = np.zeros_like(common_xs)
        for dist, weight in zip(distributions, weights):
            interp_func = interp1d(dist._xs, dist._ys, bounds_error=False, fill_value=0)
            combined_ys += weight * interp_func(common_xs)

        label = "Combined"
        for i, dist in enumerate(distributions):
            if dist.label:
                label += f"_{dist.label}"
            label += f"_{weights[i]}"

        return Distribution(common_xs, combined_ys, label=label).normalize().clean_zeros()

    def integrate(self, a=-np.inf, b=np.inf):
        """
        Integrates the distribution over a given range [a, b].

        Args:
            a (float): Lower bound of the integration range. Default is -inf.
            b (float): Upper bound of the integration range. Default is inf.

        Returns:
            float: The integral of the distribution over [a, b].
        """
        a = max(a, self._xs[0])
        b = min(b, self._xs[-1])

        mask = (self._xs >= a) & (self._xs <= b)
        return np.trapezoid(self._ys[mask], self._xs[mask])

    def mean_value(self):
        """
        Computes the mean value of the distribution.

        Returns:
            float: The mean value.
        """
        return np.trapezoid(self._xs * self._ys, self._xs)

    def __repr__(self):
        """
        Returns a string representation of the distribution, including key statistics.

        Returns:
            str: A string representation of the distribution.
        """
        label = f"{self.label}" if self.label else "Distribution"

        if Distribution.LIMIT_LABEL_LENGTH:
            label = label[: Distribution.LIMIT_LABEL_LENGTH]
            if self.label is not None and len(self.label) > Distribution.LIMIT_LABEL_LENGTH:
                label += "..."

        mean = self.mean_value()
        p1 = self.percentile(1)
        p25 = self.percentile(25)
        p50 = self.percentile(50)
        p75 = self.percentile(75)
        p95 = self.percentile(95)
        p99 = self.percentile(99)

        if not np.isclose(self.integrate(), 1):
            console.print(
                f"Warning! The integral of the distribution is not 1. Got {self.integrate()}",
                style="warning",
            )

        return f"{label}(mean={mean:.3f}, p1={p1:.3f} qs={p25:.3f}/{p50:.3f}/{p75:.3f}, p95={p95:.3f}, p99={p99:.3f})"

    def __str__(self):
        """
        Returns the string representation of the distribution.

        Returns:
            str: The string representation.
        """
        return self.__repr__()

    def plot(self, ax):
        """
        Plots the distribution on a given axis.

        Args:
            ax (matplotlib.axes.Axes): The axis to plot the distribution on.
        """
        ax.plot(*self.data, label=self.label)


@dataclass
class DistributionDescriptor:
    """
    A descriptor for a distribution, which initializes a probability density function (pdf) based on the type and data provided.

    Attributes:
        type (str): The type of the distribution. Currently, only "file" and "distribution" are supported.
        data (list): The data required to initialize the distribution. If type is "file", this should be a file path (str). If type is "distribution", this should be a Distribution object.
        pdf (Distribution, optional): The initialized probability density function. Defaults to None.

    Methods:
        __post_init__(): Initializes the pdf attribute based on the type and data provided.
    """

    type: str
    data: list

    pdf: Distribution = None

    def __post_init__(self):
        if self.type == "distribution":
            assert isinstance(self.data, Distribution)
            self.pdf = self.data
        elif self.type == "file":
            assert isinstance(self.data, str)
            self.pdf = Distribution.from_file(self.data)
        elif self.type == "constant":
            self.pdf = Distribution.dirac_delta(self.data)
        elif self.type == "script":
            from utils.dynamic_execute import dynamic_execute
            dynamic_execute(self.data)
        else:
            raise ValueError(f"Unknown distribution type: {self.type}")
