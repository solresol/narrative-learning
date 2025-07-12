#!/bin/sh

uv run export_website.py
rsync -av website/ merah.cassia.ifost.org.au:/var/www/vhosts/narrative-learning.symmachus.org/htdocs/
