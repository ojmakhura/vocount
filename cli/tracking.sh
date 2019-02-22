#!/bin/sh

# run voc-18-bd-1.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/mil -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/boosting -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/median_flow -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/tld -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/kcf -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/goturn -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/mosse -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-1.mkv -o=tracking/voc-18-bd-1/csrt -n=1 -w=1 -rx=52 -ry=309 -rw=40 -rh=29 -tr=CSRT -tp

# run voc-18-bd-2
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/mil -rx=299 -ry=187 -rw=106 -rh=60 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/boosting -rx=299 -ry=187 -rw=106 -rh=60 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/median_flow -rx=299 -ry=187 -rw=106 -rh=60 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/tld -rx=299 -ry=187 -rw=106 -rh=60 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/kcf -rx=299 -ry=187 -rw=106 -rh=60 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/goturn -rx=299 -ry=187 -rw=106 -rh=60 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/mosse -rx=299 -ry=187 -rw=106 -rh=60 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-2.mkv -o=tracking/voc-18-bd-2/csrt -rx=299 -ry=187 -rw=106 -rh=60 -tr=CSRT -tp

# run voc-18-bd-3.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/mil -rw=38 -rh=35 -rx=635 -ry=224 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/boosting -rw=38 -rh=35 -rx=635 -ry=224 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/median_flow -rw=38 -rh=35 -rx=635 -ry=224 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/tld -rw=38 -rh=35 -rx=635 -ry=224 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/kcf -rw=38 -rh=35 -rx=635 -ry=224 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/goturn -rw=38 -rh=35 -rx=635 -ry=224 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/mosse -rw=38 -rh=35 -rx=635 -ry=224 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-3.mkv -o=tracking/voc-18-bd-3/csrt -rw=38 -rh=35 -rx=635 -ry=224 -tr=CSRT -tp

# run voc-18-bd-4.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/mil -rw=65 -rh=35 -rx=13 -ry=60 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/boosting -rw=65 -rh=35 -rx=13 -ry=60 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/median_flow -rw=65 -rh=35 -rx=13 -ry=60 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/tld -rw=65 -rh=35 -rx=13 -ry=60 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/kcf -rw=65 -rh=35 -rx=13 -ry=60 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/goturn -rw=65 -rh=35 -rx=13 -ry=60 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/mosse -rw=65 -rh=35 -rx=13 -ry=60 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-4.mkv -o=tracking/voc-18-bd-4/csrt -rw=65 -rh=35 -rx=13 -ry=60 -tr=CSRT -tp

# run voc-18-bd-5.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/mil -rw=130 -rh=54 -rx=236 -ry=128 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/boosting -rw=130 -rh=54 -rx=236 -ry=128 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/median_flow -rw=130 -rh=54 -rx=236 -ry=128 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/tld -rw=130 -rh=54 -rx=236 -ry=128 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/kcf -rw=130 -rh=54 -rx=236 -ry=128 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/goturn -rw=130 -rh=54 -rx=236 -ry=128 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/mosse -rw=130 -rh=54 -rx=236 -ry=128 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-5.mkv -o=tracking/voc-18-bd-5/csrt -rw=130 -rh=54 -rx=236 -ry=128 -tr=CSRT -tp

# run voc-18-bd-6.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/mil -rw=80 -rh=41 -rx=412 -ry=61 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/boosting -rw=80 -rh=41 -rx=412 -ry=61 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/median_flow -rw=80 -rh=41 -rx=412 -ry=61 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/tld -rw=80 -rh=41 -rx=412 -ry=61 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/kcf -rw=80 -rh=41 -rx=412 -ry=61 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/goturn -rw=80 -rh=41 -rx=412 -ry=61 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/mosse -rw=80 -rh=41 -rx=412 -ry=61 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-6.mkv -o=tracking/voc-18-bd-6/csrt -rw=80 -rh=41 -rx=412 -ry=61 -tr=CSRT -tp

# run voc-18-bd-7.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/mil -rw=127 -rh=37 -rx=108 -ry=131 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/boosting -rw=127 -rh=37 -rx=108 -ry=131 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/median_flow -rw=127 -rh=37 -rx=108 -ry=131 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/tld -rw=127 -rh=37 -rx=108 -ry=131 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/kcf -rw=127 -rh=37 -rx=108 -ry=131 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/goturn -rw=127 -rh=37 -rx=108 -ry=131 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/mosse -rw=127 -rh=37 -rx=108 -ry=131 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-7.mkv -o=tracking/voc-18-bd-7/csrt -rw=127 -rh=37 -rx=108 -ry=131 -tr=CSRT -tp

# run voc-18-bd-8.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/mil -rw=79 -rh=51 -rx=324 -ry=180 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/boosting -rw=79 -rh=51 -rx=324 -ry=180 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/median_flow -rw=79 -rh=51 -rx=324 -ry=180 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/tld -rw=79 -rh=51 -rx=324 -ry=180 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/kcf -rw=79 -rh=51 -rx=324 -ry=180 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/goturn -rw=79 -rh=51 -rx=324 -ry=180 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/mosse -rw=79 -rh=51 -rx=324 -ry=180 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-8.mkv -o=tracking/voc-18-bd-8/csrt -rw=79 -rh=51 -rx=324 -ry=180 -tr=CSRT -tp

# run voc-18-bd-9.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/mil -rw=93 -rh=56 -rx=256 -ry=166 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/boosting -rw=93 -rh=56 -rx=256 -ry=166 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/median_flow -rw=93 -rh=56 -rx=256 -ry=166 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/tld -rw=93 -rh=56 -rx=256 -ry=166 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/kcf -rw=93 -rh=56 -rx=256 -ry=166 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/goturn -rw=93 -rh=56 -rx=256 -ry=166 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/mosse -rw=93 -rh=56 -rx=256 -ry=166 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-9.mkv -o=tracking/voc-18-bd-9/csrt -rw=93 -rh=56 -rx=256 -ry=166 -tr=CSRT -tp

# run voc-18-bd-10.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/mil -rw=76 -rh=83 -rx=90 -ry=254 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/boosting -rw=76 -rh=83 -rx=90 -ry=254 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/median_flow -rw=76 -rh=83 -rx=90 -ry=254 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/tld -rw=76 -rh=83 -rx=90 -ry=254 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/kcf -rw=76 -rh=83 -rx=90 -ry=254 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/goturn -rw=76 -rh=83 -rx=90 -ry=254 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/mosse -rw=76 -rh=83 -rx=90 -ry=254 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-10.mkv -o=tracking/voc-18-bd-10/csrt -rw=76 -rh=83 -rx=90 -ry=254 -tr=CSRT -tp

# run voc-18-bd-11.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/mil -rw=135 -rh=108 -rx=220 -ry=97 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/boosting -rw=135 -rh=108 -rx=220 -ry=97 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/median_flow -rw=135 -rh=108 -rx=220 -ry=97 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/tld -rw=135 -rh=108 -rx=220 -ry=97 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/kcf -rw=135 -rh=108 -rx=220 -ry=97 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/goturn -rw=135 -rh=108 -rx=220 -ry=97 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/mosse -rw=135 -rh=108 -rx=220 -ry=97 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-11.mkv -o=tracking/voc-18-bd-11/csrt -rw=135 -rh=108 -rx=220 -ry=97 -tr=CSRT -tp

# run voc-18-bd-12.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/mil -rw=21 -rh=24 -rx=439 -ry=304 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/boosting -rw=21 -rh=24 -rx=439 -ry=304 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/median_flow -rw=21 -rh=24 -rx=439 -ry=304 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/tld -rw=21 -rh=24 -rx=439 -ry=304 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/kcf -rw=21 -rh=24 -rx=439 -ry=304 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/goturn -rw=21 -rh=24 -rx=439 -ry=304 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/mosse -rw=21 -rh=24 -rx=439 -ry=304 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-12.mkv -o=tracking/voc-18-bd-12/csrt -rw=21 -rh=24 -rx=439 -ry=304 -tr=CSRT -tp

# run voc-18-bd-13.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/mil -rw=23 -rh=19 -rx=471 -ry=339 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/boosting -rw=23 -rh=19 -rx=471 -ry=339 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/median_flow -rw=23 -rh=19 -rx=471 -ry=339 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/tld -rw=23 -rh=19 -rx=471 -ry=339 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/kcf -rw=23 -rh=19 -rx=471 -ry=339 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/goturn -rw=23 -rh=19 -rx=471 -ry=339 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/mosse -rw=23 -rh=19 -rx=471 -ry=339 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-13.mkv -o=tracking/voc-18-bd-13/csrt -rw=23 -rh=19 -rx=471 -ry=339 -tr=CSRT -tp

# run voc-18-bd-14.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/mil -rw=84 -rh=60 -rx=557 -ry=337 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/boosting -rw=84 -rh=60 -rx=557 -ry=337 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/median_flow -rw=84 -rh=60 -rx=557 -ry=337 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/tld -rw=84 -rh=60 -rx=557 -ry=337 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/kcf -rw=84 -rh=60 -rx=557 -ry=337 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/goturn -rw=84 -rh=60 -rx=557 -ry=337 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/mosse -rw=84 -rh=60 -rx=557 -ry=337 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-14.mkv -o=tracking/voc-18-bd-14/csrt -rw=84 -rh=60 -rx=557 -ry=337 -tr=CSRT -tp

# run voc-18-bd-15.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/mil -rw=40 -rh=28 -rx=334 -ry=167 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/boosting -rw=40 -rh=28 -rx=334 -ry=167 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/median_flow -rw=40 -rh=28 -rx=334 -ry=167 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/tld -rw=40 -rh=28 -rx=334 -ry=167 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/kcf -rw=40 -rh=28 -rx=334 -ry=167 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/goturn -rw=40 -rh=28 -rx=334 -ry=167 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/mosse -rw=40 -rh=28 -rx=334 -ry=167 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-15.mkv -o=tracking/voc-18-bd-15/csrt -rw=40 -rh=28 -rx=334 -ry=167 -tr=CSRT -tp

# run voc-18-bd-16.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/mil -rw=111 -rh=63 -rx=48 -ry=279 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/boosting -rw=111 -rh=63 -rx=48 -ry=279 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/median_flow -rw=111 -rh=63 -rx=48 -ry=279 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/tld -rw=111 -rh=63 -rx=48 -ry=279 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/kcf -rw=111 -rh=63 -rx=48 -ry=279 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/goturn -rw=111 -rh=63 -rx=48 -ry=279 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/mosse -rw=111 -rh=63 -rx=48 -ry=279 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-16.mkv -o=tracking/voc-18-bd-16/csrt -rw=111 -rh=63 -rx=48 -ry=279 -tr=CSRT -tp

# run voc-18-bd-17.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/mil -rw=49 -rh=32 -rx=24 -ry=218 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/boosting -rw=49 -rh=32 -rx=24 -ry=218 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/median_flow -rw=49 -rh=32 -rx=24 -ry=218 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/tld -rw=49 -rh=32 -rx=24 -ry=218 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/kcf -rw=49 -rh=32 -rx=24 -ry=218 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/goturn -rw=49 -rh=32 -rx=24 -ry=218 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/mosse -rw=49 -rh=32 -rx=24 -ry=218 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-17.mkv -o=tracking/voc-18-bd-17/csrt -rw=49 -rh=32 -rx=24 -ry=218 -tr=CSRT -tp

# run voc-18-bd-18.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/mil -rw=42 -rh=39 -rx=430 -ry=311 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/boosting -rw=42 -rh=39 -rx=430 -ry=311 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/median_flow -rw=42 -rh=39 -rx=430 -ry=311 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/tld -rw=42 -rh=39 -rx=430 -ry=311 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/kcf -rw=42 -rh=39 -rx=430 -ry=311 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/goturn -rw=42 -rh=39 -rx=430 -ry=311 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/mosse -rw=42 -rh=39 -rx=430 -ry=311 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-18.mkv -o=tracking/voc-18-bd-18/csrt -rw=42 -rh=39 -rx=430 -ry=311 -tr=CSRT -tp

# run voc-18-bd-19.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/mil -rw=25 -rh=21 -rx=442 -ry=181 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/boosting -rw=25 -rh=21 -rx=442 -ry=181 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/median_flow -rw=25 -rh=21 -rx=442 -ry=181 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/tld -rw=25 -rh=21 -rx=442 -ry=181 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/kcf -rw=25 -rh=21 -rx=442 -ry=181 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/goturn -rw=25 -rh=21 -rx=442 -ry=181 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/mosse -rw=25 -rh=21 -rx=442 -ry=181 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-19.mkv -o=tracking/voc-18-bd-19/csrt -rw=25 -rh=21 -rx=442 -ry=181 -tr=CSRT -tp

# run voc-18-bd-20.mkv
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/mil -rw=121 -rh=81 -rx=312 -ry=97 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/boosting -rw=121 -rh=81 -rx=312 -ry=97 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/median_flow -rw=121 -rh=81 -rx=312 -ry=97 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/tld -rw=121 -rh=81 -rx=312 -ry=97 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/kcf -rw=121 -rh=81 -rx=312 -ry=97 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/goturn -rw=121 -rh=81 -rx=312 -ry=97 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/mosse -rw=121 -rh=81 -rx=312 -ry=97 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/birds/voc-18-bd-20.mkv -o=tracking/voc-18-bd-20/csrt -rw=121 -rh=81 -rx=312 -ry=97 -tr=CSRT -tp

# run voc-18-bl-1.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/mil -rw=20 -rh=21 -rx=569 -ry=89 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/boosting -rw=20 -rh=21 -rx=569 -ry=89 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/median_flow -rw=20 -rh=21 -rx=569 -ry=89 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/tld -rw=20 -rh=21 -rx=569 -ry=89 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/kcf -rw=20 -rh=21 -rx=569 -ry=89 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/goturn -rw=20 -rh=21 -rx=569 -ry=89 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/mosse -rw=20 -rh=21 -rx=569 -ry=89 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-1.mkv -o=tracking/voc-18-bl-1/csrt -rw=20 -rh=21 -rx=569 -ry=89 -tr=CSRT -tp

# run voc-18-bl-2.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/mil -rw=34 -rh=35 -rx=701 -ry=227 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/boosting -rw=34 -rh=35 -rx=701 -ry=227 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/median_flow -rw=34 -rh=35 -rx=701 -ry=227 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/tld -rw=34 -rh=35 -rx=701 -ry=227 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/kcf -rw=34 -rh=35 -rx=701 -ry=227 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/goturn -rw=34 -rh=35 -rx=701 -ry=227 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/mosse -rw=34 -rh=35 -rx=701 -ry=227 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-2.mkv -o=tracking/voc-18-bl-1/csrt -rw=34 -rh=35 -rx=701 -ry=227 -tr=CSRT -tp

# run voc-18-bl-3.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/mil -rw=90 -rh=89 -rx=1377 -ry=377 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/boosting -rw=90 -rh=89 -rx=1377 -ry=377 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/median_flow -rw=90 -rh=89 -rx=1377 -ry=377 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/tld -rw=90 -rh=89 -rx=1377 -ry=377 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/kcf -rw=90 -rh=89 -rx=1377 -ry=377 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/goturn -rw=90 -rh=89 -rx=1377 -ry=377 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/mosse -rw=90 -rh=89 -rx=1377 -ry=377 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-3.mkv -o=tracking/voc-18-bl-1/csrt -rw=90 -rh=89 -rx=1377 -ry=377 -tr=CSRT -tp

# run voc-18-bl-4.mkv
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/mil -rw=107 -rh=111 -rx=1208 -ry=386 -tr=MIL -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/boosting -rw=107 -rh=111 -rx=1208 -ry=386 -tr=BOOSTING -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/median_flow -rw=107 -rh=111 -rx=1208 -ry=386 -tr=MEDIAN_FLOW -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/tld -rw=107 -rh=111 -rx=1208 -ry=386 -tr=TLD -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/kcf -rw=107 -rh=111 -rx=1208 -ry=386 -tr=KCF -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/goturn -rw=107 -rh=111 -rx=1208 -ry=386 -tr=GOTURN -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/mosse -rw=107 -rh=111 -rx=1208 -ry=386 -tr=MOSSE -tp
cli/vocount_cli -v=../voc-18/videos/blood/voc-18-bl-4.mkv -o=tracking/voc-18-bl-1/csrt -rw=107 -rh=111 -rx=1208 -ry=386 -tr=CSRT -tp
