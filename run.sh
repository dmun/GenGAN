#!/bin/bash
source remote.sh

CTRL=~/.ssh/cm-%r@%h:%p

SSH_OPTS="-p $PORT \
          -o ControlMaster=auto \
          -o ControlPath=$CTRL \
          -o ControlPersist=5m"

ssh $SSH_OPTS -MNf $REMOTE

if [[ "$1" == "--pull" ]];
then
    rsync -avzP \
        -e "ssh $SSH_OPTS" \
        "$REMOTE:$DIR/audio_" .
else
    rsync -avzP \
        --exclude='data_files' \
        --filter=':- .gitignore' \
        --exclude='.git' \
        -e "ssh $SSH_OPTS" \
        . "$REMOTE:$DIR"
fi

ssh -t $SSH_OPTS $REMOTE "bash -lc 'cd $DIR && rm -rf ./logs/model3/run_0/ && uv run train.py --trial model3 --epochs 25 --batch_size 25'"
