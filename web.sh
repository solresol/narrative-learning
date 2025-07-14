#!/bin/sh

uv run export_website.py --progress && \
rsync -av website/ merah.cassia.ifost.org.au:/var/www/vhosts/narrative-learning.symmachus.org/htdocs/
