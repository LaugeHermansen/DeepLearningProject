# create infinite loop updating git repo every 30 seconds

while true; do
    git add .
    git commit -m "auto commit"
    git pull
    git push
    sleep 30
done