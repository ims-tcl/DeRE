import sys
from typing import Sequence, Iterator, TypeVar, Union, Callable

T = TypeVar("T")


def progressify(seq: Sequence[T], message: Union[str, Callable[[int, T], str]] = "") -> Iterator[T]:
    """
    Display a progress bar in the terminal while iterating over a sequence.

    This function can be used whereever we iterate over a sequence (i.e.
    something iterable with a length) and we want to display progress being
    made to the user. It works by passing on the values from the sequence (by
    yielding them), which printing an updated progress bar to the terminal. The
    progress is estimated based on how far we are through the sequence (based
    on our iteration step and its length). The progress bar is as long as its
    length, unless that is larger than the maximum length to avoid overflowing
    on small terminal sizes or large input. The maximum length is set to 30, to
    leave some space for a message, that can be printed along with the progress
    bar itself (even on old standard 80-column terminals).

    Printing the progress bar works through the use of block-drawing characters
    and carriage returns (ASCII code 0x0d). In order to avoid the cursor hiding
    part of the progress bar, we use terminal escape codes to temporarily hide
    it during the runtime of this generator.

    Args:
        seq: The sequence of elements to iterate over (often a list).
        string: An optional message that can be printed along with the bar. If it
            is a string, it will be printed, with special sequences starting with
            a percent sign replaced by various values:
                %i    Index (starting from 0)
                %e    Current element
            If it is a callable, it will be called with (index, current_element)
            as arguments, and its return value will be printed

    Yields:
        The elements from seq.

    Example:
        for element in progressify([0.01, 0.1, 0.25, 0.5, 0.9, 0.99]):
            do_something_with(element)  # progress bar is updated at each step

        for x in range(2):
            for y in progressify(range(3), lambda i, _: "step %d/6" % (x * 3 + i + 1)):
                pass # e.g. "[▓▓░] step 5/6" when x==1 and y==2
    """
    if isinstance(message, str):
        format_str = message

        def message(i: int, element: T) -> str:
            # map special sequences to special values
            replacements = {
                "%i": i,
                "%e": element,
            }
            # copies so we evaluate this at every step
            for key in replacements:
                s = format_str.replace(key, str(replacements[key]))
            return s
    assert not isinstance(message, str)
    try:
        print("\033[?25l")  # hide the cursor
        length = len(seq)
        maxlen = 30
        for i, element in enumerate(seq):
            # we copy the index (i) in order for %i to refer to the real,
            # rather than the scaled index
            if length > maxlen:
                i_chars = int(i * maxlen/length)
            else:
                i_chars = i
            print(
                "\r[{}{}] {}".format(
                    "▓"*(i_chars+1),
                    "░"*((length if length <= maxlen else maxlen)-i_chars-1),
                    message(i, element),
                ),
                file=sys.stderr,
                flush=True,  # to see the updated bar immediately
                end=""
            )
            yield element
    finally:
        print("\033[?25h")  # show the cursor again
        print()
