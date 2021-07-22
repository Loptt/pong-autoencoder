#!/bin/bash

echo "1. Images transform normal"
echo "2. Images transform big ball"
echo "3. Images transform no paddle"
echo "4. Images transform no paddle big ball"
echo "5. Images transform no ball"
echo -n "Choose a number:"

read input

case $input in
	"1")
		python3 transform_pong.py 0 $(ls images/ | wc -l) && \
		cp images_trans/* images_trans/inputs
		;;
	"2")
		python3 transform_pong_big.py 0 $(ls images/ | wc -l) && \
		cp images_big/* images_big/inputs
		;;
	"3")
		python3 transform_pong_paddleless.py 0 $(ls images/ | wc -l) && \
		cp images_paddleless/* images_paddleless/inputs
		;;
	"4")
		python3 transform_pong_paddleless_big.py 0 $(ls images/ | wc -l) && \
		cp images_paddleless_big/* images_paddleless_big/inputs
		;;
	"5")
		python3 transform_pong_ballless.py 0 $(ls images/ | wc -l) && \
		cp images_ballless/* images_ballless/inputs
		;;
esac

echo "Done"
