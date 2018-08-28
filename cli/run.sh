#!/bin/sh

# run voc-18-bd-1.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/unextended -n=1 -w=1 -s -d -f -df -rx=52 -ry=309 -rw=40 -rh=29
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/extended -n=1 -w=1 -s -d -f -df -rx=52 -ry=309 -rw=40 -rh=29 -e

# run voc-18-bd-2.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/unextended -n=1 -w=1 -s -d -f -df -rx=299 -ry=187 -rw=106 -rh=60
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/extended -n=1 -w=1 -s -d -f -df -rx=299 -ry=187 -rw=106 -rh=60 -e

# run voc-18-bd-3.mkv
#cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/unextended -n=1 -w=1 -s -d -f -df -rw=40 -rh=31 -rx=131, -ry=314
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/unextended -n=1 -w=1 -s -d -f -df -rw=43 -rh=29 -rx=130, -ry=314
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/extended -n=1 -w=1 -s -d -f -df -rw=43 -rh=29 -rx=130, -ry=314 -e

# run voc-18-bd-4.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-4 -o=out/voc-18-bd-4/unextended -n=1 -w=1 -s -d -f -df -rw=65 -rh=35 -rx=13 -ry=60
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-4 -o=out/voc-18-bd-4/extended -n=1 -w=1 -s -d -f -df -rw=65 -rh=35 -rx=13 -ry=60 -e

# run voc-18-bd-5.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/unextended -n=1 -w=1 -s -d -f -df -rw=130 -rh=54 -rx=236 -ry=128
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/extended -n=1 -w=1 -s -d -f -df -rw=130 -rh=54 -rx=236 -ry=128 -e

# run voc-18-bd-6.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-6 -o=out/voc-18-bd-6/unextended -n=1 -w=1 -s -d -f -df -rw=80 -rh=41 -rx=412 -ry=61
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-6 -o=out/voc-18-bd-6/extended -n=1 -w=1 -s -d -f -df -rw=80 -rh=41 -rx=412 -ry=61 -e

# run voc-18-bd-7.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/unextended -n=1 -w=1 -s -d -f -df -rw=127 -rh=37 -rx=108 -ry=131
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/extended -n=1 -w=1 -s -d -f -df -rw=127 -rh=37 -rx=108 -ry=131 -e

# run voc-18-bd-8.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-8 -o=out/voc-18-bd-8/unextended -n=1 -w=1 -s -d -f -df -rw=79 -rh=51 -rx=324 -ry=180
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-8 -o=out/voc-18-bd-8/extended -n=1 -w=1 -s -d -f -df -rw=79 -rh=51 -rx=324 -ry=180 -e

# run voc-18-bd-9.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-9 -o=out/voc-18-bd-9/unextended -n=1 -w=1 -s -d -f -df -rw=93 -rh=56 -rx=256 -ry=166
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-9 -o=out/voc-18-bd-9/extended -n=1 -w=1 -s -d -f -df -rw=93 -rh=56 -rx=256 -ry=166 -e

# run voc-18-bd-10.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/unextended -n=1 -w=1 -s -d -f -df -rw=76 -rh=83 -rx=90 -ry=254
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/extended -n=1 -w=1 -s -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -e

# run voc-18-bd-11.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-11 -o=out/voc-18-bd-11/unextended -n=1 -w=1 -s -d -f -df -rw=135 -rh=108 -rx=220 -ry=97
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-11 -o=out/voc-18-bd-11/extended -n=1 -w=1 -s -d -f -df -rw=135 -rh=108 -rx=220 -ry=97 -e

# run voc-18-bd-12.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-12 -o=out/voc-18-bd-12/unextended -n=1 -w=1 -s -d -f -df -rw=27 -rh=23 -rx=252 -ry=285
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-12 -o=out/voc-18-bd-12/extended -n=1 -w=1 -s -d -f -df -rw=27 -rh=23 -rx=252 -ry=285 -e

# run voc-18-bd-13.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-13 -o=out/voc-18-bd-13/unextended -n=1 -w=1 -s -d -f -df -rw=23 -rh=19 -rx=471 -ry=339
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-13 -o=out/voc-18-bd-13/extended -n=1 -w=1 -s -d -f -df -rw=23 -rh=19 -rx=471 -ry=339 -e

# run voc-18-bd-14.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-14 -o=out/voc-18-bd-14/unextended -n=1 -w=1 -s -d -f -df -rw=84 -rh=60 -rx=557 -ry=337
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-14 -o=out/voc-18-bd-14/extended -n=1 -w=1 -s -d -f -df -rw=84 -rh=60 -rx=557 -ry=337 -e

# run voc-18-bd-15.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/unextended -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/extended -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -e

# run voc-18-bd-16.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-16 -o=out/voc-18-bd-16/unextended -n=1 -w=1 -s -d -f -df -rw=111 -rh=63 -rx=48 -ry=279
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-16 -o=out/voc-18-bd-16/extended -n=1 -w=1 -s -d -f -df -rw=111 -rh=63 -rx=48 -ry=279 -e

# run voc-18-bd-17.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/unextended -n=1 -w=1 -s -d -f -df -rw=64 -rh=38 -rx=233 -ry=266
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/extended -n=1 -w=1 -s -d -f -df -rw=64 -rh=38 -rx=233 -ry=266 -e

# run voc-18-bd-18.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-18 -o=out/voc-18-bd-18/unextended -n=1 -w=1 -s -d -f -df -rw=42 -rh=39 -rx=430 -ry=311
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-18 -o=out/voc-18-bd-18/extended -n=1 -w=1 -s -d -f -df -rw=42 -rh=39 -rx=430 -ry=311 -e

# run voc-18-bd-19.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-19 -o=out/voc-18-bd-19/unextended -n=1 -w=1 -s -d -f -df -rw=25 -rh=21 -rx=442 -ry=181
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-19 -o=out/voc-18-bd-19/extended -n=1 -w=1 -s -d -f -df -rw=25 -rh=21 -rx=442 -ry=181 -e

# run voc-18-bd-20.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-20 -o=out/voc-18-bd-20/unextended -n=1 -w=1 -s -d -f -df -rw=121 -rh=81 -rx=312 -ry=97
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-20 -o=out/voc-18-bd-20/extended -n=1 -w=1 -s -d -f -df -rw=121 -rh=81 -rx=312 -ry=97 -e

# run voc-18-bl-1.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/unextended -n=1 -w=1 -s -d -f -df -rw=20 -rh=21 -rx=569 -ry=89
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/extended -n=1 -w=1 -s -d -f -df -rw=20 -rh=21 -rx=569 -ry=89 -e

cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/unextended_rotated -r -n=1 -w=1 -s -d -f -df -rw=20 -rh=21 -rx=569 -ry=89
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/extended_rotated -r -n=1 -w=1 -s -d -f -df -rw=20 -rh=21 -rx=569 -ry=89 -e

# run voc-18-bl-2.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/unextended -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/extended -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -e

cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/unextended_rotated -r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/extended_rotated -r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -e

# run voc-18-bl-3.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/unextended -n=1 -w=1 -s -d -f -df -rw=90 -rh=89 -rx=1377 -ry=377
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/extended -n=1 -w=1 -s -d -f -df -rw=90 -rh=89 -rx=1377 -ry=377 -e

cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/unextended_rotated -r -n=1 -w=1 -s -d -f -df -rw=90 -rh=89 -rx=1377 -ry=377
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/extended_rotated -r -n=1 -w=1 -s -d -f -df -rw=90 -rh=89 -rx=1377 -ry=377 -e

# run voc-18-bl-4.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-4 -o=out/voc-18-bl-4/unextended -n=1 -w=1 -s -d -f -df -rw=107 -rh=111 -rx=1208 -ry=386
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-4 -o=out/voc-18-bl-4/extended -n=1 -w=1 -s -d -f -df -rw=107 -rh=111 -rx=1208 -ry=386 -e

cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-4 -o=out/voc-18-bl-4/unextended_rotated -r -n=1 -w=1 -s -d -f -df -rw=107 -rh=111 -rx=1208 -ry=386
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-4 -o=out/voc-18-bl-4/extended_rotated -r -n=1 -w=1 -s -d -f -df -rw=107 -rh=111 -rx=1208 -ry=386 -e
