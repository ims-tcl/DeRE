"""
This script checks all xml files in the specs folder against the dere.dtd file.
"""

import re
import os.path
import codecs
from lxml import etree
from typing import Optional

dtdRe = re.compile(".*<!DOCTYPE .* [\"'](.*\.dtd)[\"']>.*")
theDtd = "../data/frame-specifications/xmlv1/dere.dtd"


def validate_all() -> None:
    for inFile in os.listdir():
        if inFile.split(".")[1] == "xml":
            print("-" * 80)
            print("Checking " + inFile)
            fdir = os.path.abspath(os.path.dirname(inFile))
            with codecs.open(inFile, "r", "utf-8") as inf:
                for ln in inf:
                    mtch = dtdRe.match(ln)
                    if mtch:
                        if os.path.isabs(mtch.group(1)):
                            theDtd = mtch.group(1)
                        else:
                            theDtd = os.path.abspath(fdir + "/" + mtch.group(1))
                        break

            print("Using DTD:", theDtd)

            parser = etree.XMLParser(dtd_validation=True)
            dtd = etree.DTD(open(theDtd))
            tree = etree.parse(inFile)

            valid = dtd.validate(tree)
            if valid:
                print("XML was valid!")

            else:
                print("XML was not valid:")
                print(dtd.error_log.filter_from_errors())


if __name__ == "__main__":
    validate_all()
