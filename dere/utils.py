import sys

def progressify(ls, string=""):
    try:
        print("\033[?25l")
        length = len(ls)
        maxlen = 30
        for i, element in enumerate(ls):
            if length > maxlen:
                i = int(i * maxlen/length)
            print(
                "\r[{}{}] {}".format(
                    "▓"*(i+1),
                    "░"*((length if length <= maxlen else maxlen)-i-1),
                    string.replace("%i", str(element)),
                ),
                file=sys.stderr,
                flush=True,
                end=""
            )
            yield element
    finally:
        print("\033[?25h")
        print()
