#!/bin/sh

# run voc-18-bd-1.mkv
console/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/unextended -n=1 -w=1 -s -d -f -df -e -rx=52 -ry=309 -rw=40 -rh=29
console/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/extended -n=1 -w=1 -s -d -f -df -e -rx=52 -ry=309 -rw=40 -rh=29 -e

# run voc-18-bd-2.mkv
console/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/unextended -n=1 -w=1 -s -d -f -df -rx=299 -ry=187 -rw=106 -rh=60
console/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/extended -n=1 -w=1 -s -d -f -df -rx=299 -ry=187 -rw=106 -rh=60 -e
