#!/usr/bin/env bash
set -e

if [[ $# -eq 2 ]]
then
  extra="--worker_idx $1 --world_size $2"
fi

[ -z "${TXT_ROOT}" ] && exit
[ -z "${IMG_ROOT}" ] && exit
[ -z "${SPA_ROOT}" ] && exit

for SPLIT in "test" "train" "validation"
do
  echo $SPLIT
  TXT_SRC="${TXT_ROOT}/${SPLIT}"
  IMG_SRC="${IMG_ROOT}/${SPLIT}"
  SPA_DST="${SPA_ROOT}/${SPLIT}"
  python sparse_from_realestate_format.py --txt_src $TXT_SRC --img_src $IMG_SRC --spa_dst $SPA_DST $extra
done
