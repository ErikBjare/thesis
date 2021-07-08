# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Convert thesis tex files to rst
import subprocess
from pathlib import Path

subprocess.call(Path(__file__).parent.parent / "tex-to-rst.sh")


# -- Project information -----------------------------------------------------

project = "thesis"
copyright = "2021, Erik Bjäreholt"
author = "Erik Bjäreholt"


# -- General configuration ---------------------------------------------------

notebooks = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.bibtex",
] + (["myst_nb"] if notebooks else ["myst_parser"])

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Timeout for cells in notebooks built by myst_nb
execution_timeout = 180

bibtex_bibfiles = ["../../tex/zotero-bibtex.bib", "../../tex/misc.bib"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# References

import pybtex.plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
    join,
    sentence,
    tag,
    words,
    first_of,
    optional,
    field,
    optional_field,
    optional,
)
from pybtex.richtext import Text, Tag, Symbol
import re


def dashify(text):
    dash_re = re.compile(r"-+")
    return Text(Symbol("ndash")).join(text.split(dash_re))


pages = field("pages", apply_func=dashify)
date = field("date")


# https://bitbucket.org/pybtex-devs/pybtex/src/HEAD/pybtex/style/formatting/unsrt.py
class MyStyle(UnsrtStyle):
    """Needed to work around missing 'journal' keys in BibLaTeX files exported by Zotero"""

    # def format_article(self, e):
    #     entry = e["entry"]
    #     print(entry)
    #     return Text("Article ", Tag("em", entry.fields["title"]))
    #     if "volume" in entry:
    #         volume_and_pages = join[field("volume"), optional[":", pages]]
    #     else:
    #         volume_and_pages = words["pages", optional[pages]]
    #     template = toplevel[
    #         self.format_names("author"),
    #         sentence[field("title")],
    #         sentence[tag("emph")[field("journaltitle")], volume_and_pages, date],
    #     ]
    #     return template.format_data(entry)

    def format_software(self, e):
        entry = e["entry"]
        return Text("Software ", Tag("em", entry.fields["title"]))

    # def get_article_template(self, e):
    #     volume_and_pages = first_of[
    #         # volume and pages, with optional issue number
    #         optional[
    #             join[field("volume"), optional["(", field("number"), ")"], ":", pages],
    #         ],
    #         # pages only
    #         words["pages", pages],
    #     ]
    #     template = toplevel[
    #         self.format_names("author"),
    #         self.format_title(e, "title"),
    #         sentence[
    #             tag("em")[field("journaltitle")], optional[volume_and_pages], date
    #         ],
    #         sentence[optional_field("note")],
    #         self.format_web_refs(e),
    #     ]
    #     return template


pybtex.plugin.register_plugin("pybtex.style.formatting", "mystyle", MyStyle)

bibtex_default_style = "mystyle"
