#!/bin/sh
# This is a comment!

for file in sample/*.png;
do
echo $file
python VDTv1.0.py --file $file
done 

#python VDTv1.0.py  --lamb 0.01 --file 4K_pt3deg_180_IR_rec00000102.png 
