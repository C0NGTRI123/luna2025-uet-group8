#!/bin/bash

set -eux

ruff check app

typos app
