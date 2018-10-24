#!/bin/sh

# run voc-18-bd-1.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/3_0 -n=1 -w=1 -d -f -df -rx=52 -ry=309 -rw=40 -rh=29
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/3_2 -n=1 -w=1 -d -f -df -rx=52 -ry=309 -rw=40 -rh=29 -I=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/3_1_O -n=1 -w=1 -d -f -df -rx=52 -ry=309 -rw=40 -rh=29 -I=1 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/2_1 -n=1 -w=1 -d -f -df -rx=52 -ry=309 -rw=40 -rh=29 -minPts=2 -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/2_2 -n=1 -w=1 -d -f -df -rx=52 -ry=309 -rw=40 -rh=29 -minPts=2 -I=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-1 -o=out/voc-18-bd-1/2_3 -n=1 -w=1 -d -f -df -rx=52 -ry=309 -rw=40 -rh=29 -minPts=2 -I=3

# run voc-18-bd-2.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/3_0 -n=1 -w=1 -d -f -df -rx=299 -ry=187 -rw=106 -rh=60
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/2_0 -n=1 -w=1 -d -f -df -rx=299 -ry=187 -rw=106 -rh=60 -minPts=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-2 -o=out/voc-18-bd-2/2_0_r -n=1 -w=1 -d -f -df -rx=299 -ry=187 -rw=106 -rh=60 -minPts=2 -r

# run voc-18-bd-3.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/3_0 -n=1 -w=1 -d -f -df -rw=38 -rh=35 -rx=635 -ry=224
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/3_0_r -n=1 -w=1 -d -f -df -rw=38 -rh=35 -rx=635 -ry=224 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/3_1_r -n=1 -w=1 -d -f -df -rw=38 -rh=35 -rx=635 -ry=224 -r -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/3_1_O_r -n=1 -w=1 -d -f -df -rw=38 -rh=35 -rx=635 -ry=224 -r -I=1 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-3 -o=out/voc-18-bd-3/2_3_r -n=1 -w=1 -d -rw=38 -rh=35 -rx=635 -ry=224 -r -minPts=2 -I=3

# run voc-18-bd-4.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-4 -o=out/voc-18-bd-4/3_0 -n=1 -w=1 -d -f -df -rw=65 -rh=35 -rx=13 -ry=60

# run voc-18-bd-5.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/3_0 -n=1 -w=1 -d -f -df -rw=130 -rh=54 -rx=236 -ry=128
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/3_0_r -n=1 -w=1 -d -f -df -rw=130 -rh=54 -rx=236 -ry=128 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/3_0_O -n=1 -w=1 -d -f -df -rw=130 -rh=54 -rx=236 -ry=128 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/3_0_O_r -n=1 -w=1 -d -f -df -rw=130 -rh=54 -rx=236 -ry=128 -O -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/2_0 -n=1 -w=1 -d -f -df -rw=130 -rh=54 -rx=236 -ry=128 -minPts=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-5 -o=out/voc-18-bd-5/2_0_r -n=1 -w=1 -d -f -df -rw=130 -rh=54 -rx=236 -ry=128 -minPts=2 -r

# run voc-18-bd-6.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-6 -o=out/voc-18-bd-6/3_0 -n=1 -w=1 -d -f -df -rw=80 -rh=41 -rx=412 -ry=61

# run voc-18-bd-7.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/3_0 -n=1 -w=1 -d -f -df -rw=127 -rh=37 -rx=108 -ry=131
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/3_0_r -n=1 -w=1 -d -f -df -rw=127 -rh=37 -rx=108 -ry=131 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/3_0_O_r -n=1 -w=1 -d -f -df -rw=127 -rh=37 -rx=108 -ry=131 -r -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/2_0_r -n=1 -w=1 -d -f -df -rw=127 -rh=37 -rx=108 -ry=131 -minPts=2 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-7 -o=out/voc-18-bd-7/2_1_r -n=1 -w=1 -d -f -df -rw=127 -rh=37 -rx=108 -ry=131 -minPts=2 -r -I=1

# run voc-18-bd-8.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-8 -o=out/voc-18-bd-8/3_0 -n=1 -w=1 -d -f -df -rw=79 -rh=51 -rx=324 -ry=180

# run voc-18-bd-9.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-9 -o=out/voc-18-bd-9/3_0 -n=1 -w=1 -d -rw=93 -rh=56 -rx=256 -ry=166
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-9 -o=out/voc-18-bd-9/3_0_r -n=1 -w=1 -d -rw=93 -rh=56 -rx=256 -ry=166 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-9 -o=out/voc-18-bd-9/2_0 -n=1 -w=1 -d -rw=93 -rh=56 -rx=256 -ry=166 -minPts=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-9 -o=out/voc-18-bd-9/2_0_r -n=1 -w=1 -d -rw=93 -rh=56 -rx=256 -ry=166 -minPts=2 -r

# run voc-18-bd-10.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/3_0 -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/3_0_r -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/3_0_O -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/3_0_O_r -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -O -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/2_0 -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -minPts=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/2_0_r -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -minPts=2 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/2_1 -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -minPts=2 -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-10 -o=out/voc-18-bd-10/2_1_r -n=1 -w=1 -d -f -df -rw=76 -rh=83 -rx=90 -ry=254 -minPts=2 -r -I=1

# run voc-18-bd-11.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-11 -o=out/voc-18-bd-11/3_0 -n=1 -w=1 -d -f -df -rw=135 -rh=108 -rx=220 -ry=97

# run voc-18-bd-12.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-12 -o=out/voc-18-bd-12/3_0 -n=1 -w=1 -d -f -df -rw=21 -rh=24 -rx=439 -ry=304
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-12 -o=out/voc-18-bd-12/3_0_r -n=1 -w=1 -d -f -df -rw=21 -rh=24 -rx=439 -ry=304 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-12 -o=out/voc-18-bd-12/3_0_O -n=1 -w=1 -d -f -df -rw=21 -rh=24 -rx=439 -ry=304 -O

# run voc-18-bd-13.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-13 -o=out/voc-18-bd-13/3_0 -n=1 -w=1 -d -f -df -rw=23 -rh=19 -rx=471 -ry=339
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-13 -o=out/voc-18-bd-13/3_0_r -n=1 -w=1 -d -f -df -rw=23 -rh=19 -rx=471 -ry=339 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-13 -o=out/voc-18-bd-13/3_0_O -n=1 -w=1 -d -f -df -rw=23 -rh=19 -rx=471 -ry=339 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-13 -o=out/voc-18-bd-13/3_0_O_r -n=1 -w=1 -d -f -df -rw=23 -rh=19 -rx=471 -ry=339 -O -r

# run voc-18-bd-14.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-14 -o=out/voc-18-bd-14/3_0 -n=1 -w=1 -s -d -f -df -rw=84 -rh=60 -rx=557 -ry=337

# run voc-18-bd-15.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/3_0 -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/3_0_r -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/3_1 -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/3_1_r -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -r -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/3_0_O -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/2_0 -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -minPts=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/2_1 -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -minPts=2 -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/2_2 -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -minPts=2 -I=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-15 -o=out/voc-18-bd-15/2_2_r -n=1 -w=1 -s -d -f -df -rw=52 -rh=35 -rx=200 -ry=249 -minPts=2 -I=2 -r

# run voc-18-bd-16.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-16 -o=out/voc-18-bd-16/3_0 -n=1 -w=1 -s -d -f -df -rw=111 -rh=63 -rx=48 -ry=279

# run voc-18-bd-17.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/3_0 -n=1 -w=1 -s -d -f -df -rw=33 -rh=30 -rx=339 -ry=227
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/3_0_r -n=1 -w=1 -s -d -f -df -rw=33 -rh=30 -rx=339 -ry=227 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/3_1 -n=1 -w=1 -s -d -f -df -rw=33 -rh=30 -rx=339 -ry=227 -I=1
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/3_1_r -n=1 -w=1 -s -d -f -df -rw=33 -rh=30 -rx=339 -ry=227 -I=1 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/3_0_O -n=1 -w=1 -s -d -f -df -rw=33 -rh=30 -rx=339 -ry=227 -O
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-17 -o=out/voc-18-bd-17/3_0_O_r -n=1 -w=1 -s -d -f -df -rw=33 -rh=30 -rx=339 -ry=227 -O -r

# run voc-18-bd-18.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-18 -o=out/voc-18-bd-18/3_0 -n=1 -w=1 -s -d -f -df -rw=42 -rh=39 -rx=430 -ry=311

# run voc-18-bd-19.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-19 -o=out/voc-18-bd-19/3_0 -n=1 -w=1 -s -d -f -df -rw=25 -rh=21 -rx=442 -ry=181

# run voc-18-bd-20.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-20 -o=out/voc-18-bd-20/3_0 -n=1 -w=1 -s -d -f -df -rw=121 -rh=81 -rx=312 -ry=97
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-20 -o=out/voc-18-bd-20/3_0_r -n=1 -w=1 -s -d -f -df -rw=121 -rh=81 -rx=312 -ry=97 -r
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-20 -o=out/voc-18-bd-20/2_0 -n=1 -w=1 -s -d -f -df -rw=121 -rh=81 -rx=312 -ry=97 -minPts=2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -t=../voc-18/truth-lmdb/birds/voc-18-bd-20 -o=out/voc-18-bd-20/2_0_r -n=1 -w=1 -s -d -f -df -rw=121 -rh=81 -rx=312 -ry=97 -minPts=2 -r

# run voc-18-bl-1.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/3_0 -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/3_0_r -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -r
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/3_0_O -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -O
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/3_0_O_r -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -O -r
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/3_1_O -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -I=1 -O
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/3_1_O_r -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -r -I=1 -O
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/2_1 -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -minPts=2 -I=1
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/2_1_r -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -r -minPts=2 -I=1
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/2_3 -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -minPts=2 -I=3
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/2_3_r -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -r -minPts=2 -I=3
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/2_4 -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -minPts=2 -I=4
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-1 -o=out/voc-18-bl-1/2_4_r -n=1 -w=1 -s -d -rw=20 -rh=21 -rx=569 -ry=89 -r -minPts=2 -I=4
 
# run voc-18-bl-2.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/3_1_r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -r -I=1
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/3_2_r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -r -I=2
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/2_2_r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -r -I=2 -minPts=2
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/2_3_r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -r -I=3 -minPts=2
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-2 -o=out/voc-18-bl-2/2_4_r -n=1 -w=1 -s -d -f -df -rw=34 -rh=35 -rx=701 -ry=227 -r -I=4 -minPts=2

# run voc-18-bl-3.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_0 -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_0_r -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -r
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_0_O -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -O
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_0_O_r -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -r -O
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_1 -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -I=1
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_1_r -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -r -I=1
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_1_O -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -O -I=1
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-3 -o=out/voc-18-bl-3/3_1_O_r -n=1 -w=1 -s -d -rw=90 -rh=89 -rx=1377 -ry=377 -r -O -I=1

# run voc-18-bl-4.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-4 -o=out/voc-18-bl-4/2_2 -n=1 -w=1 -s -d -rw=107 -rh=111 -rx=1208 -ry=386 -I=2 -minPts=2
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -t=../voc-18/truth-lmdb/blood/voc-18-bl-4 -o=out/voc-18-bl-4/2_2_r -n=1 -w=1 -s -d -rw=107 -rh=111 -rx=1208 -ry=386 -I=2 -r -minPts=2

