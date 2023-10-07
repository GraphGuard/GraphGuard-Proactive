# shellcheck disable=SC2006
py_path=`which python`
dataset=$1
number=$2
model=$3
device=$4
# shellcheck disable=SC2006
run() {
    shift
    for i in $(seq $number); do
      # shellcheck disable=SC2068
      $@
      "$py_path" Proactive_MIA_node_level_revise.py --config "config/$dataset.json" --model "$model" --device "$device"
    done
}

# shellcheck disable=SC2046
# shellcheck disable=SC2006
#echo $epoch
run