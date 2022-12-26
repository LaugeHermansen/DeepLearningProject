#!/bin/bash

while true; do
  git add .
  git commit -m "Automatic commit"
  git pull
  git push
  sleep 120
done
