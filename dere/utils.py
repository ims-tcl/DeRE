import sys
from typing import Sequence, Generator, TypeVar

T = TypeVar("T")

def progressify(seq: Sequence[T], string: str = "") -> Generator[T, None, None]:
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
        string: An optional message that can be printed along with the bar. It
            can contain the special sequence "%i", which will be replaced with
            a string representation of the current item.

    Yields:
        The elements from seq.

    Example:
        for element in progressify([0.01, 0.1, 0.25, 0.5, 0.9, 0.99]):
            do_something_with(element)  # progress bar is updated at each step
    """
    try:
        print("\033[?25l")  # hide the cursor
        length = len(seq)
        maxlen = 30
        for i, element in enumerate(seq):
            if length > maxlen:
                i = int(i * maxlen/length)
            print(
                "\r[{}{}] {}".format(
                    "▓"*(i+1),
                    "░"*((length if length <= maxlen else maxlen)-i-1),
                    string.replace("%i", str(element)),
                ),
                file=sys.stderr,
                flush=True,  # to see the updated bar immediately
                end=""
            )
            yield element
    finally:
        print("\033[?25h")  # show the cursor again
        print()
