#!/bin/bash
# Run this script as  `./update.sh <app>` to rebuild the app:

docker-compose -f docker/$1/docker-compose.yaml build
